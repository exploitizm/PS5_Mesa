/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_MESH_COMPUTE_H
#define RADV_MESH_COMPUTE_H

#include <stdbool.h>
#include <stdint.h>

#include "radv_shader.h"

/* Mesh shaders run as compute on the PS5 (docs/MESH_COMPUTE_DESIGN.md in EglantinePS5). The
 * GPU's NGG mesh launch (GFX10.1, a vertex a mesh workgroup in GS mode) deadlocks primitive
 * assembly and the shader exporter in a way that depends on what ran before, so a pipeline with
 * a mesh shader, with an emulated task shader (radv_task_emulated) or none, runs it as a compute
 * pass that writes
 * each mesh workgroup's outputs to a slot in memory, and rasterizes them through a vertex shader
 * that reads them. Per emulated task chunk: the task shader; the records pass
 * (radv_meta_mesh_compute_records: a record a task workgroup, with the mesh workgroups before it);
 * without a task shader, the records pass over the draws' own grids (a record a draw);
 * then per pass of up to capacity mesh workgroups, the compute pass (each workgroup finds its
 * record by binary search) and a triangle-list draw of 3 vertices per primitive slot. Passes
 * past the chunk's mesh workgroups are skipped by the command processor (COND_EXEC on the pass's
 * workgroup count).
 *
 * The device's buffer (RADV_MESH_COMPUTE_BYTES): a header, then the slots.
 *
 * The header: the first mesh workgroup of the current pass (written by the command processor
 * before each pass), the chunk's records and mesh workgroups (written by the records pass), and
 * each pass's dispatch (x, 1, 1, 0) and copy draw (vertices, 1, 0, 0).
 *
 * A slot (stride bytes, one per mesh workgroup of a pass, at radv_mesh_compute_slots_va): the
 * vertex and primitive counts (two dwords, 16 bytes with padding), then max_vertices x the
 * per-vertex outputs (16 bytes each, in the order of their varying slots), then max_primitives
 * triangles (their three vertex indices, a byte each, in a dword). */

/* The slots: a pass holds as many mesh workgroups as fit, so a mesh shader with large outputs
 * runs more passes, which cost nothing when empty. */
#define RADV_MESH_COMPUTE_SLOTS_BYTES    (16u << 20)
#define RADV_MESH_COMPUTE_HEADER_BYTES   16384u
#define RADV_MESH_COMPUTE_BYTES          (RADV_MESH_COMPUTE_HEADER_BYTES + RADV_MESH_COMPUTE_SLOTS_BYTES)
#define RADV_MESH_COMPUTE_HEADER_BASE    0u  /* the pass's first mesh workgroup */
#define RADV_MESH_COMPUTE_HEADER_RECORDS 4u  /* the chunk's records, then its mesh workgroups */
#define RADV_MESH_COMPUTE_HEADER_PASSES  64u /* a pass's dispatch, then its draw */
#define RADV_MESH_COMPUTE_PASS_BYTES     32u
#define RADV_MESH_COMPUTE_MAX_PASSES                                                                \
   ((RADV_MESH_COMPUTE_HEADER_BYTES - RADV_MESH_COMPUTE_HEADER_PASSES) / RADV_MESH_COMPUTE_PASS_BYTES)
/* The most a pass's packets take (radv_mesh_compute_chunk), reserved so a skipped pass is one
 * span of the command buffer. */
#define RADV_MESH_COMPUTE_PASS_DWORDS 2048u
/* The mesh workgroups an emulated task chunk may have before the rest are left out. */
#define RADV_MESH_COMPUTE_CHUNK_WORKGROUPS 65536u
/* Mesh pipelines this path takes (part of the cache's UUID, radv_device_get_cache_uuid): 2 adds
 * those without a task shader, cached before on the hardware path. */
#define RADV_MESH_COMPUTE_VERSION 2u

struct radv_compiler_info;
struct radv_graphics_state_key;

/* Where a draw's buffers are, for the compute pass and the copy: uploaded per draw, and named by
 * the shaders' gs_compute_draw argument (AC_UD_CS_GS_COMPUTE_DRAW, AC_UD_VS_GS_COMPUTE_DRAW), so
 * the shaders hold no address and their pipelines can be cached. */
struct radv_mesh_compute_block {
   uint64_t records; /* the task emulation's records (radv_mesh_draw_record) */
   uint64_t payload; /* its task payload ring */
   uint64_t header;  /* the device's buffer */
   uint64_t slots;   /* its slots, after the header */
};

/* Whether mesh shaders take this path: unless /data/eglantineps5/radv-mesh-compute.txt says 0
 * (read once). */
bool radv_mesh_compute_enabled(void);

struct radv_device;
/* The device's buffer, made once (the first pipeline with a mesh shader). */
VkResult radv_mesh_compute_init(struct radv_device *device);

struct radv_mesh_compute_split {
   /* The mesh stage as it was (its layout and key compile the compute pass). */
   struct radv_shader_stage mesh_stage;
   /* The mesh shader, lowered to the compute pass. */
   nir_shader *cs;
   /* The copy draw's vertices a mesh workgroup: max primitives x 3. */
   uint32_t vertices_per_workgroup;
   /* The mesh workgroups a pass holds. */
   uint32_t capacity;
};

bool radv_mesh_compute_wanted(const struct radv_graphics_state_key *gfx_state,
                              const struct radv_shader_stage *stages);

/* stages[MESA_SHADER_MESH] becomes the compute pass (out->cs) and stages[MESA_SHADER_VERTEX] the
 * copy vertex shader; the mesh stage is left empty. */
void radv_mesh_compute_split(const struct radv_compiler_info *compiler_info, struct radv_shader_stage *stages,
                             struct radv_mesh_compute_split *out);

/* The block's address in a shader of this path, once its arguments are declared (the copy vertex
 * shader's in the pipeline compile, the compute pass's in radv_mesh_compute_compile). */
bool radv_mesh_compute_lower_block(nir_shader *nir, const struct radv_shader_stage *stage);

struct radv_shader_binary *radv_mesh_compute_compile(const struct radv_compiler_info *compiler_info,
                                                     struct radv_mesh_compute_split *split,
                                                     struct radv_shader_debug_info *debug);

#endif /* RADV_MESH_COMPUTE_H */
