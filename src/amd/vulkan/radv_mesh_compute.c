/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_mesh_compute.h"
#include "nir/radv_nir.h"
#include "nir_builder.h"
#include "radv_constants.h"
#include "radv_buffer.h"
#include "radv_device.h"
#include "radv_pipeline.h"
#include "radv_shader.h"
#include "radv_shader_args.h"
#include "radv_shader_info.h"
#include "ac_nir.h"
#include "util/bitscan.h"
#include "util/u_math.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"

#include <stdio.h>
#include <stdlib.h>

bool
radv_mesh_compute_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      int value = 1;
      FILE *f = fopen("/data/eglantineps5/radv-mesh-compute.txt", "r");
      if (f) {
         char text[8] = {0};
         if (fread(text, 1, sizeof(text) - 1, f) > 0 && text[0] == '0')
            value = 0;
         fclose(f);
      }
      if (!value)
         fprintf(stderr, "radv/ps5: mesh shaders run on the hardware mesh path (radv-mesh-compute.txt)\n");
      p_atomic_set(&enabled, value);
   }
   return enabled;
}

VkResult
radv_mesh_compute_init(struct radv_device *device)
{
   static simple_mtx_t lock = SIMPLE_MTX_INITIALIZER;
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&lock);
   if (!device->mesh_compute_bo) {
      struct radeon_winsys_bo *bo;
      result = radv_bo_create(device, NULL, RADV_MESH_COMPUTE_BYTES, 4096, RADEON_DOMAIN_VRAM,
                              RADEON_FLAG_NO_CPU_ACCESS | RADEON_FLAG_NO_INTERPROCESS_SHARING,
                              RADV_BO_PRIORITY_SCRATCH, 0, true, &bo);
      if (result == VK_SUCCESS) {
         result = device->ws->buffer_make_resident(device->ws, bo, true);
         if (result != VK_SUCCESS) {
            radv_bo_destroy(device, NULL, bo);
         } else {
            p_atomic_set(&device->mesh_compute_bo, bo);
         }
      }
   }
   simple_mtx_unlock(&lock);
   return result;
}

/* Where a mesh workgroup's outputs go (radv_mesh_compute.h). */
struct mesh_compute_layout {
   uint64_t vtx_mask; /* per-vertex outputs, in slot order */
   unsigned num_vtx_slots;
   unsigned max_vertices;
   unsigned max_primitives;
   unsigned prim_offset; /* the triangles, after the vertices */
   unsigned stride;      /* a slot */
   unsigned capacity;    /* slots in a pass */
   unsigned payload_stride;
   uint32_t address32_hi; /* of the block (radv_mesh_compute_block) */
};

/* The block's address until the shader's arguments are declared (radv_mesh_compute_lower_block). */
#define BLOCK_ARG_PLACEHOLDER 0xffffu

/* A 64-bit field of the draw's block (radv_mesh_compute_block). */
static nir_def *
block_field(nir_builder *b, const struct mesh_compute_layout *l, unsigned offset)
{
   nir_def *arg = nir_load_scalar_arg_amd(b, 1, .base = BLOCK_ARG_PLACEHOLDER);
   nir_def *block = nir_pack_64_2x32_split(b, arg, nir_imm_int(b, l->address32_hi));
   return nir_pack_64_2x32(b, ac_nir_load_smem(b, 2, block, nir_imm_int(b, offset), 8, ACCESS_CAN_SPECULATE));
}

static bool
lower_block_arg(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct radv_shader_args *args = data;
   if (intr->intrinsic != nir_intrinsic_load_scalar_arg_amd || nir_intrinsic_base(intr) != BLOCK_ARG_PLACEHOLDER)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, ac_nir_load_arg(b, &args->ac, args->gs_compute_draw));
   return true;
}

bool
radv_mesh_compute_lower_block(nir_shader *nir, const struct radv_shader_stage *stage)
{
   bool progress = false;
   assert(stage->args.gs_compute_draw.used);
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_block_arg, nir_metadata_control_flow,
            (void *)&stage->args);
   return progress;
}

static const uint64_t allowed_vertex_outputs = VARYING_BIT_POS | BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32);

/* Why a mesh shader keeps the hardware path, or NULL when it runs as compute. */
static const char *
mesh_compute_refusal(const struct radv_graphics_state_key *gfx_state, const struct radv_shader_stage *stages)
{
   const nir_shader *mesh = stages[MESA_SHADER_MESH].nir;
   const nir_shader *fs = stages[MESA_SHADER_FRAGMENT].nir;

   if (mesh->info.mesh.primitive_type != MESA_PRIM_TRIANGLES)
      return "not triangles";
   if (gfx_state->has_multiview_view_index)
      return "multiview";
   if (!mesh->info.mesh.max_vertices_out || mesh->info.mesh.max_vertices_out > 256 ||
       !mesh->info.mesh.max_primitives_out || mesh->info.mesh.max_primitives_out > 256)
      return "more than 256 vertices or primitives";

   const uint64_t written = mesh->info.outputs_written;
   const uint64_t per_prim = mesh->info.per_primitive_outputs;
   if ((written & ~per_prim) & ~allowed_vertex_outputs)
      return "a per-vertex output other than the position and generic varyings";
   if ((written & per_prim) != VARYING_BIT_PRIMITIVE_INDICES)
      return "per-primitive outputs";
   if (!(written & VARYING_BIT_POS))
      return "no position";

   if (fs) {
      nir_foreach_shader_in_variable (var, fs) {
         if (var->data.per_primitive || var->data.location == VARYING_SLOT_PRIMITIVE_ID ||
             var->data.location == VARYING_SLOT_LAYER || var->data.location == VARYING_SLOT_VIEWPORT)
            return "a per-primitive fragment input";
      }
   }
   return NULL;
}

bool
radv_mesh_compute_wanted(const struct radv_graphics_state_key *gfx_state, const struct radv_shader_stage *stages)
{
   /* Mesh shaders with an emulated task shader, and without one (DRAW_INDEX_AUTO of a vertex a
    * workgroup, the same NGG launch). */
   if (!stages[MESA_SHADER_MESH].nir || !radv_mesh_compute_enabled())
      return false;

   const char *refusal = mesh_compute_refusal(gfx_state, stages);
   if (refusal) {
      static unsigned told;
      if (p_atomic_inc_return(&told) <= 16)
         fprintf(stderr, "radv/ps5 mesh-compute: a mesh shader%s keeps the hardware path: %s\n",
                 stages[MESA_SHADER_TASK].nir ? " with a task shader" : "", refusal);
      return false;
   }
   return true;
}

static unsigned
vertex_slot(const struct mesh_compute_layout *l, unsigned location)
{
   return util_bitcount64(l->vtx_mask & BITFIELD64_MASK(location));
}

/* The copy vertex shader: vertex v draws corner v % 3 of primitive (v / 3) % max_primitives of
 * mesh workgroup v / (3 max_primitives); a primitive past its workgroup's count is a triangle of
 * three equal corners, which the rasterizer drops. */
static nir_shader *
build_copy_vs(const nir_shader_compiler_options *options, const struct mesh_compute_layout *l, unsigned wave_size)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, options, "radv_mesh_compute_copy");
   /* The subgroup size a vertex stage gets from SPIR-V (radv_shader_spirv_to_nir): RADV's wave
    * size is the NIR's (radv_nir_shader_info_pass), and the builder's 1 to 128 had ACO compile
    * for waves of 1 and spill without end. */
   b.shader->info.min_subgroup_size = wave_size;
   b.shader->info.max_subgroup_size = wave_size;
   b.shader->info.api_subgroup_size = wave_size;
   const unsigned vpw = l->max_primitives * 3;

   nir_def *v = nir_load_vertex_id_zero_base(&b);
   nir_def *wg = nir_udiv_imm(&b, v, vpw);
   nir_def *rem = nir_umod_imm(&b, v, vpw);
   nir_def *prim = nir_udiv_imm(&b, rem, 3);
   nir_def *corner = nir_umod_imm(&b, rem, 3);
   nir_def *in_buffer = nir_ult_imm(&b, wg, l->capacity);
   wg = nir_umin(&b, wg, nir_imm_int(&b, l->capacity - 1));

   nir_def *slot = nir_iadd(&b, block_field(&b, l, offsetof(struct radv_mesh_compute_block, slots)),
                            nir_u2u64(&b, nir_imul_imm(&b, wg, l->stride)));
   nir_def *counts = nir_build_load_global(&b, 2, 32, slot, .align_mul = 8);
   nir_def *num_vtx = nir_umin(&b, nir_channel(&b, counts, 0), nir_imm_int(&b, l->max_vertices));
   nir_def *num_prm = nir_umin(&b, nir_channel(&b, counts, 1), nir_imm_int(&b, l->max_primitives));
   nir_def *valid = nir_iand(&b, in_buffer, nir_iand(&b, nir_ult(&b, prim, num_prm), nir_ine_imm(&b, num_vtx, 0)));

   nir_def *tri = nir_build_load_global(
      &b, 1, 32,
      nir_iadd(&b, slot, nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, prim, 4), l->prim_offset))),
      .align_mul = 4);
   nir_def *index = nir_ubfe(&b, tri, nir_imul_imm(&b, corner, 8), nir_imm_int(&b, 8));
   index = nir_umin(&b, index, nir_iadd_imm(&b, nir_umax(&b, num_vtx, nir_imm_int(&b, 1)), -1));
   /* An unused primitive slot: all three corners on vertex 0. */
   index = nir_bcsel(&b, valid, index, nir_imm_int(&b, 0));

   u_foreach_bit64 (location, l->vtx_mask) {
      const unsigned s = vertex_slot(l, location);
      nir_def *addr = nir_iadd(
         &b, slot,
         nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, nir_iadd_imm(&b, nir_imul_imm(&b, index, l->num_vtx_slots), s), 16), 16)));
      nir_def *value = nir_build_load_global(&b, 4, 32, addr, .align_mul = 16);
      if (location == VARYING_SLOT_POS) {
         /* All three corners of an unused slot at one point: no area. */
         value = nir_bcsel(&b, valid, value, nir_imm_vec4(&b, 0.0, 0.0, 0.0, 1.0));
      }
      nir_variable *var = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "mesh_output");
      var->data.location = location;
      nir_store_var(&b, var, value, 0xf);
   }

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

struct lower_state {
   const struct mesh_compute_layout *l;
   nir_instr *raw_workgroup_id;
   nir_def *slot;
   nir_def *grid;
   nir_def *index; /* the workgroup's index in its grid */
   nir_def *entry; /* its task workgroup's ring entry */
   nir_def *draw_id;
   nir_def *payload; /* the task payload ring */
};

static nir_def *
vertex_output_addr(nir_builder *b, struct lower_state *s, nir_intrinsic_instr *intr)
{
   const struct mesh_compute_layout *l = s->l;
   const unsigned location = nir_intrinsic_io_semantics(intr).location;
   nir_def *vertex = nir_umin(b, nir_get_io_arrayed_index_src(intr)->ssa, nir_imm_int(b, l->max_vertices - 1));
   nir_def *slot_index = nir_iadd_imm(b, nir_get_io_offset_src(intr)->ssa, vertex_slot(l, location));
   nir_def *dword = nir_iadd_imm(b, nir_imul_imm(b, nir_iadd(b, nir_imul_imm(b, vertex, l->num_vtx_slots), slot_index), 4),
                                 nir_intrinsic_component(intr));
   return nir_iadd(b, s->slot, nir_u2u64(b, nir_iadd_imm(b, nir_imul_imm(b, dword, 4), 16)));
}

static nir_def *
primitive_byte_addr(nir_builder *b, struct lower_state *s, nir_intrinsic_instr *intr, unsigned component)
{
   const struct mesh_compute_layout *l = s->l;
   nir_def *prim = nir_umin(b, nir_get_io_arrayed_index_src(intr)->ssa, nir_imm_int(b, l->max_primitives - 1));
   nir_def *byte = nir_iadd_imm(b, nir_imul_imm(b, prim, 4), l->prim_offset + component);
   return nir_iadd(b, s->slot, nir_u2u64(b, byte));
}

static bool
lower_mesh_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_state *s = data;
   const struct mesh_compute_layout *l = s->l;

   if (&intr->instr == s->raw_workgroup_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_workgroup_id: {
      nir_def *gx = nir_umax(b, nir_channel(b, s->grid, 0), nir_imm_int(b, 1));
      nir_def *gy = nir_umax(b, nir_channel(b, s->grid, 1), nir_imm_int(b, 1));
      nir_def *x = nir_umod(b, s->index, gx);
      nir_def *y = nir_umod(b, nir_udiv(b, s->index, gx), gy);
      nir_def *z = nir_udiv(b, s->index, nir_imul(b, gx, gy));
      nir_def_replace(&intr->def, nir_vec3(b, x, y, z));
      return true;
   }
   case nir_intrinsic_load_workgroup_index:
      nir_def_replace(&intr->def, s->index);
      return true;
   case nir_intrinsic_load_num_workgroups:
      nir_def_replace(&intr->def, s->grid);
      return true;
   case nir_intrinsic_load_draw_id:
      nir_def_replace(&intr->def, s->draw_id);
      return true;
   case nir_intrinsic_load_view_index:
      nir_def_replace(&intr->def, nir_imm_int(b, 0));
      return true;
   case nir_intrinsic_load_task_payload: {
      nir_def *base = nir_iadd_imm(b, nir_imul_imm(b, s->entry, l->payload_stride), nir_intrinsic_base(intr));
      nir_def *addr = nir_iadd(b, s->payload, nir_u2u64(b, nir_iadd(b, base, intr->src[0].ssa)));
      nir_def *value = nir_build_load_global(b, intr->def.num_components, intr->def.bit_size, addr, .align_mul = 4);
      nir_def_replace(&intr->def, value);
      return true;
   }
   case nir_intrinsic_set_vertex_and_primitive_count: {
      nir_if *if_first = nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      nir_build_store_global(b, nir_vec2(b, intr->src[0].ssa, intr->src[1].ssa), s->slot, .align_mul = 8);
      nir_pop_if(b, if_first);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_store_per_vertex_output: {
      nir_def *value = intr->src[0].ssa;
      if (value->bit_size != 32)
         value = nir_u2u32(b, value);
      nir_build_store_global(b, value, vertex_output_addr(b, s, intr), .write_mask = nir_intrinsic_write_mask(intr),
                             .align_mul = 4);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_load_per_vertex_output: {
      nir_def *value = nir_build_load_global(b, intr->def.num_components, 32, vertex_output_addr(b, s, intr),
                                             .align_mul = 4);
      if (intr->def.bit_size != 32)
         value = nir_u2uN(b, value, intr->def.bit_size);
      nir_def_replace(&intr->def, value);
      return true;
   }
   case nir_intrinsic_store_per_primitive_output: {
      if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PRIMITIVE_INDICES)
         return false;
      nir_def *value = intr->src[0].ssa;
      const unsigned first = nir_intrinsic_component(intr);
      u_foreach_bit (c, nir_intrinsic_write_mask(intr))
         nir_build_store_global(b, nir_u2u8(b, nir_channel(b, value, c)), primitive_byte_addr(b, s, intr, first + c),
                                .align_mul = 1);
      nir_instr_remove(&intr->instr);
      return true;
   }
   case nir_intrinsic_load_per_primitive_output: {
      if (nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PRIMITIVE_INDICES)
         return false;
      const unsigned first = nir_intrinsic_component(intr);
      nir_def *comps[4];
      for (unsigned c = 0; c < intr->def.num_components; c++)
         comps[c] = nir_u2uN(b, nir_build_load_global(b, 1, 8, primitive_byte_addr(b, s, intr, first + c), .align_mul = 1),
                             intr->def.bit_size);
      nir_def_replace(&intr->def, nir_vec(b, comps, intr->def.num_components));
      return true;
   }
   case nir_intrinsic_barrier: {
      nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (modes & nir_var_shader_out) {
         nir_intrinsic_set_memory_modes(intr, (modes & ~nir_var_shader_out) | nir_var_mem_global);
         return true;
      }
      return false;
   }
   default:
      return false;
   }
}

static void
lower_mesh_to_compute(nir_shader *nir, const struct mesh_compute_layout *l)
{
   radv_nir_lower_io(nir);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   b.impl = impl;

   /* The workgroup's mesh workgroup in its chunk: the pass's first, and its own in the pass. */
   nir_def *raw = nir_load_workgroup_id(&b);
   struct lower_state s = {.l = l, .raw_workgroup_id = nir_def_instr(raw)};
   nir_def *wg = nir_channel(&b, raw, 0);
   nir_def *header = block_field(&b, l, offsetof(struct radv_mesh_compute_block, header));
   nir_def *base = nir_build_load_global(&b, 1, 32, nir_iadd_imm(&b, header, RADV_MESH_COMPUTE_HEADER_BASE),
                                         .align_mul = 4, .access = ACCESS_COHERENT);
   nir_def *records = nir_build_load_global(&b, 1, 32, nir_iadd_imm(&b, header, RADV_MESH_COMPUTE_HEADER_RECORDS),
                                            .align_mul = 4, .access = ACCESS_COHERENT);
   nir_def *g = nir_iadd(&b, base, wg);

   /* Its task workgroup's record (radv_mesh_draw_record, a task workgroup each, in order): the last
    * whose first mesh workgroup is not past it. The records' index is the task workgroup's ring
    * entry. */
   nir_def *records_va = block_field(&b, l, offsetof(struct radv_mesh_compute_block, records));
   s.payload = block_field(&b, l, offsetof(struct radv_mesh_compute_block, payload));
   nir_variable *lo_var = nir_local_variable_create(b.impl, glsl_uint_type(), "record_lo");
   nir_variable *hi_var = nir_local_variable_create(b.impl, glsl_uint_type(), "record_hi");
   nir_store_var(&b, lo_var, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, hi_var, nir_umax(&b, records, nir_imm_int(&b, 1)), 0x1);
   nir_push_loop(&b);
   {
      nir_def *lo = nir_load_var(&b, lo_var);
      nir_def *hi = nir_load_var(&b, hi_var);
      nir_break_if(&b, nir_uge(&b, nir_iadd_imm(&b, lo, 1), hi));
      nir_def *mid = nir_ushr_imm(&b, nir_iadd(&b, lo, hi), 1);
      nir_def *first = nir_build_load_global(
         &b, 1, 32,
         nir_iadd(&b, records_va,
                  nir_u2u64(&b, nir_iadd_imm(&b, nir_imul_imm(&b, mid, sizeof(struct radv_mesh_draw_record)),
                                             offsetof(struct radv_mesh_draw_record, first_workgroup)))),
         .align_mul = 4);
      nir_def *at_or_before = nir_uge(&b, g, first);
      nir_store_var(&b, lo_var, nir_bcsel(&b, at_or_before, mid, lo), 0x1);
      nir_store_var(&b, hi_var, nir_bcsel(&b, at_or_before, hi, mid), 0x1);
   }
   nir_pop_loop(&b, NULL);
   s.entry = nir_load_var(&b, lo_var);

   nir_def *record = nir_iadd(&b, records_va, nir_u2u64(&b, nir_imul_imm(&b, s.entry, sizeof(struct radv_mesh_draw_record))));
   s.grid = nir_build_load_global(&b, 3, 32, nir_iadd_imm(&b, record, offsetof(struct radv_mesh_draw_record, grid)),
                                  .align_mul = 4);
   nir_def *first = nir_build_load_global(&b, 1, 32,
                                          nir_iadd_imm(&b, record, offsetof(struct radv_mesh_draw_record, first_workgroup)),
                                          .align_mul = 4);
   s.index = nir_isub(&b, g, first);
   s.draw_id = nir_build_load_global(&b, 1, 32,
                                     nir_iadd_imm(&b, record, offsetof(struct radv_mesh_draw_record, task_draw_id)),
                                     .align_mul = 4);
   nir_def *slot_index = nir_umin(&b, wg, nir_imm_int(&b, l->capacity - 1));
   s.slot = nir_iadd(&b, block_field(&b, l, offsetof(struct radv_mesh_compute_block, slots)),
                     nir_u2u64(&b, nir_imul_imm(&b, slot_index, l->stride)));

   /* No counts until the shader sets them: a workgroup that never does draws nothing. */
   nir_if *if_first = nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_build_store_global(&b, nir_imm_ivec2(&b, 0, 0), s.slot, .align_mul = 8);
   nir_pop_if(&b, if_first);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_mesh_intrinsic, nir_metadata_none, &s);
   NIR_PASS(_, nir, nir_opt_dce);
}

void
radv_mesh_compute_split(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                        struct radv_mesh_compute_split *out)
{
   struct radv_shader_stage *ms = &stages[MESA_SHADER_MESH];
   nir_shader *mesh = ms->nir;

   struct mesh_compute_layout l = {0};
   l.vtx_mask = mesh->info.outputs_written & ~mesh->info.per_primitive_outputs & allowed_vertex_outputs;
   l.num_vtx_slots = util_bitcount64(l.vtx_mask);
   l.max_vertices = mesh->info.mesh.max_vertices_out;
   l.max_primitives = mesh->info.mesh.max_primitives_out;
   l.prim_offset = 16 + l.max_vertices * l.num_vtx_slots * 16;
   l.stride = align(l.prim_offset + l.max_primitives * 4, 64);
   l.capacity = MIN2(RADV_MESH_COMPUTE_SLOTS_BYTES / l.stride, RADV_MESH_COMPUTE_CHUNK_WORKGROUPS);
   l.payload_stride =
      stages[MESA_SHADER_TASK].nir ? align(MAX2(stages[MESA_SHADER_TASK].nir->info.task_payload_size, 16), 16) : 16;
   l.address32_hi = compiler_info->hw.address32_hi;

   static unsigned told;
   if (p_atomic_inc_return(&told) <= 16)
      fprintf(stderr,
              "radv/ps5 mesh-compute: a slot of %u bytes (%u vertices of %u outputs, %u primitives), %u a pass%s\n",
              l.stride, l.max_vertices, l.num_vtx_slots, l.max_primitives, l.capacity,
              stages[MESA_SHADER_TASK].nir ? "" : ", no task shader");

   nir_shader *vs = build_copy_vs(mesh->options, &l, compiler_info->key.ge_wave_size);
   lower_mesh_to_compute(mesh, &l);

   memset(out, 0, sizeof(*out));
   out->mesh_stage = *ms;
   out->cs = mesh;
   out->vertices_per_workgroup = l.max_primitives * 3;
   out->capacity = l.capacity;

   struct radv_shader_stage *vs_stage = &stages[MESA_SHADER_VERTEX];
   *vs_stage = *ms;
   vs_stage->stage = MESA_SHADER_VERTEX;
   vs_stage->nir = vs;
   vs_stage->next_stage = MESA_SHADER_FRAGMENT;
   /* Its block argument, as a geometry shader's rasterization copy has (RADV_GS_COMPUTE_RAST). */
   vs_stage->gs_compute_rast = true;

   ms->nir = NULL;
   ms->stage = MESA_SHADER_NONE;
}


static const nir_lower_compute_system_values_options compute_sysval_options = {
   .lower_local_invocation_index = true,
   .has_base_global_invocation_id = false,
   .has_base_workgroup_id = false,
};

struct radv_shader_binary *
radv_mesh_compute_compile(const struct radv_compiler_info *compiler_info, struct radv_mesh_compute_split *split,
                          struct radv_shader_debug_info *debug)
{
   nir_shader *nir = split->cs;
   struct radv_shader_stage stage;
   memset(&stage, 0, sizeof(stage));
   stage.stage = MESA_SHADER_COMPUTE;
   stage.next_stage = MESA_SHADER_NONE;
   stage.entrypoint = "main";
   stage.layout = split->mesh_stage.layout;
   stage.key = split->mesh_stage.key;
   stage.nir = nir;

   /* A compute shader of the mesh shader's workgroup, in waves of 32 as the NGG mesh shader ran. */
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.outputs_written = 0;
   nir->info.outputs_read = 0;
   nir->info.per_primitive_outputs = 0;
   nir->info.workgroup_size_variable = false;
   nir->info.min_subgroup_size = 32;
   nir->info.max_subgroup_size = 32;
   NIR_PASS(_, nir, nir_lower_compute_system_values, &compute_sysval_options);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, ac_nir_lower_indirect_derefs);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   radv_optimize_nir(nir, stage.key.optimisations_disabled);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   radv_nir_shader_info_init(MESA_SHADER_COMPUTE, MESA_SHADER_NONE, &stage.info);
   radv_nir_shader_info_pass(compiler_info, nir, &stage.layout, &stage.key, NULL, RADV_PIPELINE_COMPUTE, false,
                             &stage.info);
   /* Its block argument (radv_mesh_compute_block). */
   stage.info.gs_compute = RADV_GS_COMPUTE_MESH;
   /* A graphics stage still, as radv_gs_compute_compile's passes are. */
   if (stage.info.loads_dynamic_offsets && stage.layout.independent_sets)
      stage.info.loads_dynamic_descriptors_offset_addr = true;

   radv_declare_shader_args(compiler_info, NULL, &stage, MESA_SHADER_NONE, debug);
   stage.info.user_sgprs_locs = stage.args.user_sgprs_locs;
   stage.info.inline_push_constant_mask = stage.args.ac.inline_push_const_mask;

   radv_mesh_compute_lower_block(nir, &stage);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &compute_sysval_options);
   radv_optimize_nir(nir, stage.key.optimisations_disabled);
   radv_postprocess_nir(compiler_info, NULL, &stage);

   struct radv_shader_binary *binary = radv_shader_nir_to_asm(compiler_info, &stage, &nir, 1, NULL);
   radv_shader_dump_asm(compiler_info, debug, binary, &stage.info);
   ralloc_free(nir);
   split->cs = NULL;
   return binary;
}
