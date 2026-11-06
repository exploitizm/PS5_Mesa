/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_ps5_winsys.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "addrlib/inc/addrtypes.h"
#include "addrlib/src/amdgpu_asic_addr.h"
#include "tools/radv_debug.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "ac_linux_drm.h"
#include "sid.h"
#if defined(__PROSPERO__)
#include <ps5platform/kernel.h> /* ProsperoEden: tools/patch-radv-submission-dump.py */
#endif

/* ------------------------------------------------------------------ queue */

static struct radv_ps5_queue radv_ps5_queue_storage;
static void radv_ps5_offer_stall_dump(void);
static struct radv_ps5_queue *radv_ps5_queue_instance;
static once_flag radv_ps5_queue_once = ONCE_FLAG_INIT;

static void
radv_ps5_queue_create(void)
{
   struct radv_ps5_queue *const queue = &radv_ps5_queue_storage;
   if (!radv_ps5_platform_init())
      return;
   /* The words the GPU runs and the marker it writes live in the window, where
    * ps5vk's submission buffer has always been. */
   if (!radv_ps5_memory_alloc(RADV_PS5_RING_BYTES, RADV_PS5_LARGE_BYTES, true, &queue->ring))
      return;
   if (!radv_ps5_memory_alloc(RADV_PS5_PAGE_BYTES, RADV_PS5_PAGE_BYTES, true, &queue->marker_memory)) {
      radv_ps5_memory_free(&queue->ring);
      return;
   }
   queue->marker = (volatile uint32_t *)queue->marker_memory.cpu;
   *queue->marker = 0;
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   simple_mtx_init(&queue->submit_lock, mtx_plain);
   mtx_init(&queue->sync_lock, mtx_plain);
   cnd_init(&queue->sync_cond);
   radv_ps5_queue_instance = queue;
   radv_ps5_offer_stall_dump();
}

struct radv_ps5_queue *
radv_ps5_queue_get(void)
{
   call_once(&radv_ps5_queue_once, radv_ps5_queue_create);
   return radv_ps5_queue_instance;
}

uint64_t
radv_ps5_queue_poll(struct radv_ps5_queue *queue)
{
   const uint64_t submitted = p_atomic_read(&queue->submitted_seq);
   /* The marker holds the low 32 bits of the newest sequence the GPU has run.
    * Its CPU cache line may hold an older value, so it is evicted before the
    * read, as ps5vk's marker poll does. */
   radv_ps5_cpu_flush((const void *)queue->marker, sizeof(*queue->marker));
   const uint32_t marker = *queue->marker;
   /* The GPU never lags the CPU by 2^32 submissions, so the newest sequence
    * with these low bits that is not past the submitted one is the one. */
   const uint64_t completed = submitted - (uint32_t)((uint32_t)submitted - marker);
   uint64_t known = p_atomic_read(&queue->completed_seq);
   while (completed > known) {
      const uint64_t seen = p_atomic_cmpxchg(&queue->completed_seq, known, completed);
      if (seen == known) {
         known = completed;
         break;
      }
      known = seen;
   }
   return known;
}

/* A marker that makes no progress for this long while work is outstanding
 * means the GPU is not running it: the device is lost (ps5vk gives up after 2 s;
 * a conformance run's heaviest cases need more). */
#define RADV_PS5_STALL_NS (UINT64_C(10) * 1000000000)
/* Before sleeping, the marker is checked in a loop for this long: most waits
 * end well inside a millisecond (PS5_Vulkan R36). */
#define RADV_PS5_SPIN_NS UINT64_C(1500000)

/* ProsperoEden (tools/patch-radv-submission-dump.py): the 32-bit value at a GPU address, as the
 * CPU reads it now. A buffer's GPU address is its CPU address (radv_ps5_bo.c), but the address may
 * belong to a buffer since freed, so the kernel is asked first whether it is mapped for the CPU;
 * the line is then flushed from the CPU cache, as the marker poll does, so the read sees memory.
 * Only the stall dump calls it. */
static bool
radv_ps5_stall_read(uint64_t address, uint32_t *value)
{
#if defined(__PROSPERO__)
   if (address % 4 || address >= RADV_PS5_GPU_ADDRESS_LIMIT)
      return false;
   void *start = NULL, *end = NULL;
   uint32_t protection = 0;
   if (sceKernelQueryMemoryProtection((void *)(uintptr_t)address, &start, &end, &protection) != 0 ||
       (uintptr_t)start > address || (uintptr_t)end < address + sizeof(uint32_t) ||
       !(protection & PS5_KERNEL_PROT_CPU_READ))
      return false;
   radv_ps5_cpu_flush((const void *)(uintptr_t)address, sizeof(uint32_t));
   *value = *(const volatile uint32_t *)(uintptr_t)address;
   return true;
#else
   /* The host model runs no GPU, so nothing stalls. */
   (void)address;
   (void)value;
   return false;
#endif
}

/* ProsperoEden (tools/patch-radv-submission-dump.py): when a submission's words stop making
 * progress for RADV_PS5_STALL_NS, say what the stuck submission and the ones queued behind it
 * carry, before the console is gone. Nothing here runs until the ten-second wait trips, but for
 * radv_ps5_record_submission (a few stores per submission) and radv_ps5_note_shader (a few per
 * shader, at pipeline creation). */

static const char *
radv_ps5_pkt3_name(uint32_t op)
{
   switch (op) {
   case PKT3_NOP: return "NOP";
   case PKT3_DRAW_INDEX_AUTO: return "DRAW_INDEX_AUTO";
   case PKT3_DRAW_INDEX_2: return "DRAW_INDEX_2";
   case PKT3_DRAW_INDEX_OFFSET_2: return "DRAW_INDEX_OFFSET_2";
   case PKT3_DRAW_INDEX_MULTI_AUTO: return "DRAW_INDEX_MULTI_AUTO";
   case PKT3_DRAW_INDIRECT: return "DRAW_INDIRECT";
   case PKT3_DRAW_INDEX_INDIRECT: return "DRAW_INDEX_INDIRECT";
   case PKT3_DRAW_INDIRECT_MULTI: return "DRAW_INDIRECT_MULTI";
   case PKT3_DRAW_INDEX_INDIRECT_MULTI: return "DRAW_INDEX_INDIRECT_MULTI";
   case PKT3_DISPATCH_DIRECT: return "DISPATCH_DIRECT";
   case PKT3_DISPATCH_INDIRECT: return "DISPATCH_INDIRECT";
   case PKT3_WAIT_REG_MEM: return "WAIT_REG_MEM";
   case PKT3_WAIT_ON_CE_COUNTER: return "WAIT_ON_CE_COUNTER";
   case PKT3_PFP_SYNC_ME: return "PFP_SYNC_ME";
   case PKT3_COND_EXEC: return "COND_EXEC";
   case PKT3_COND_WRITE: return "COND_WRITE";
   case PKT3_INDIRECT_BUFFER: return "INDIRECT_BUFFER";
   case PKT3_RELEASE_MEM: return "RELEASE_MEM";
   case PKT3_EVENT_WRITE: return "EVENT_WRITE";
   case PKT3_ACQUIRE_MEM: return "ACQUIRE_MEM";
   case PKT3_DMA_DATA: return "DMA_DATA";
   case PKT3_WRITE_DATA: return "WRITE_DATA";
   default: return NULL;
   }
}

static bool
radv_ps5_pkt3_is_draw(uint32_t op)
{
   switch (op) {
   case PKT3_DRAW_INDEX_AUTO:
   case PKT3_DRAW_INDEX_2:
   case PKT3_DRAW_INDEX_OFFSET_2:
   case PKT3_DRAW_INDEX_MULTI_AUTO:
   case PKT3_DRAW_INDIRECT:
   case PKT3_DRAW_INDEX_INDIRECT:
   case PKT3_DRAW_INDIRECT_MULTI:
   case PKT3_DRAW_INDEX_INDIRECT_MULTI:
      return true;
   default:
      return false;
   }
}

struct radv_ps5_stream_scan {
   uint32_t packets;
   uint32_t draws;
   uint32_t dispatches;
   uint32_t waits;
   uint32_t indirect;
   uint32_t conditional;
   uint32_t last_op[6];
   uint32_t last_count;
   bool ends_with_release_mem;
   bool truncated;
};

/* Walks the submission's PM4 words, counting packets and the kinds the stall hypotheses turn on,
 * and remembering the last few opcodes. Bounded by count; the NOP padding this winsys emits is
 * one header-only word (PKT3_NOP_PAD, count 0x3fff). */
static void
radv_ps5_scan_stream(const uint32_t *words, uint32_t count, struct radv_ps5_stream_scan *scan)
{
   memset(scan, 0, sizeof(*scan));
   uint32_t last = UINT32_MAX;
   for (uint32_t i = 0; i < count;) {
      const uint32_t header = words[i];
      const uint32_t type = PKT_TYPE_G(header);
      if (type == 2) { /* a type-2 filler word is one dword */
         i += 1;
         continue;
      }
      if (type != 3) {
         scan->truncated = true;
         break;
      }
      const uint32_t op = PKT3_IT_OPCODE_G(header);
      const uint32_t body = PKT_COUNT_G(header);
      const uint32_t n = (op == PKT3_NOP && body == 0x3fff) ? 1 : body + 2;
      scan->packets++;
      last = op;
      if (scan->last_count < 6) {
         scan->last_op[scan->last_count++] = op;
      } else {
         memmove(scan->last_op, scan->last_op + 1, 5 * sizeof(uint32_t));
         scan->last_op[5] = op;
      }
      if (radv_ps5_pkt3_is_draw(op))
         scan->draws++;
      switch (op) {
      case PKT3_DRAW_INDIRECT:
      case PKT3_DRAW_INDEX_INDIRECT:
      case PKT3_DRAW_INDIRECT_MULTI:
      case PKT3_DRAW_INDEX_INDIRECT_MULTI:
      case PKT3_INDIRECT_BUFFER: scan->indirect++; break;
      case PKT3_DISPATCH_DIRECT: scan->dispatches++; break;
      case PKT3_DISPATCH_INDIRECT: scan->dispatches++; scan->indirect++; break;
      case PKT3_WAIT_REG_MEM: scan->waits++; break;
      case PKT3_COND_EXEC:
      case PKT3_COND_WRITE: scan->conditional++; break;
      default: break;
      }
      if (i + n > count) {
         scan->truncated = true;
         break;
      }
      i += n;
   }
   scan->ends_with_release_mem = last == PKT3_RELEASE_MEM;
}

/* The words a record holds, and whether they lie inside the buffer they claim: a record caught
 * mid-claim must not send a walk out of bounds. */
static bool
radv_ps5_record_words(const struct radv_ps5_queue *queue, const struct radv_ps5_in_flight *record,
                      const uint32_t **words, uint32_t *count)
{
   const bool dedicated = record->dedicated.cpu != NULL;
   uint32_t n = record->words;
   if (!n && !dedicated)
      n = record->end - record->start;
   *words = dedicated ? (const uint32_t *)record->dedicated.cpu : (const uint32_t *)queue->ring.cpu + record->start;
   *count = n;
   if (!n || n > RADV_PS5_RING_WORDS)
      return false;
   if (dedicated)
      return (uint64_t)n * sizeof(uint32_t) <= record->dedicated.bytes;
   return record->start < RADV_PS5_RING_WORDS && record->start + n <= RADV_PS5_RING_WORDS;
}

static void
radv_ps5_dump_in_flight(const struct radv_ps5_queue *queue, const struct radv_ps5_in_flight *record, uint64_t completed)
{
   const uint32_t *words;
   uint32_t count;
   const bool scanned = radv_ps5_record_words(queue, record, &words, &count);
   const char *state = record->seq <= completed    ? "done"
                       : record->seq == completed + 1 ? "STUCK"
                                                      : "queued";
   struct radv_ps5_stream_scan scan;
   if (scanned)
      radv_ps5_scan_stream(words, count, &scan);
   fprintf(stderr,
           "radv/ps5 stall:   seq %" PRIu64 " %-6s dwords=%u agc_ibs=%u queue=%u waits=%u signals=%u depends=%" PRIu64
           " %s",
           record->seq, state, count, record->parts, record->queue_index, record->wait_count, record->signal_count,
           record->depends, record->dedicated.cpu ? "dedicated" : "ring");
   if (scanned) {
      fprintf(stderr, " packets=%u draws=%u dispatches=%u gpu_waits=%u indirect=%u cond=%u completion=%s%s last:",
              scan.packets, scan.draws, scan.dispatches, scan.waits, scan.indirect, scan.conditional,
              scan.ends_with_release_mem ? "ok" : "MISSING", scan.truncated ? " TRUNCATED" : "");
      for (uint32_t i = 0; i < scan.last_count; i++) {
         const char *name = radv_ps5_pkt3_name(scan.last_op[i]);
         if (name)
            fprintf(stderr, " %s", name);
         else
            fprintf(stderr, " 0x%02x", scan.last_op[i]);
      }
   } else {
      fprintf(stderr, " (words out of range, not scanned)");
   }
   fprintf(stderr, "\n");
}

/* ------------------------------------------------- whose code sits at a GPU address */

/* Every shader pipeline creation made and every vertex-input prolog, newest last, in a ring that
 * overwrites its oldest entries once full. Writers claim a slot with one atomic increment; the
 * reader is the stall dump, ten seconds after anything was written. */
#define RADV_PS5_SHADER_NOTES 65536

struct radv_ps5_shader_note {
   uint64_t va;
   uint64_t pipeline_hash;
   uint64_t pipeline;
   const char *stage;
   uint32_t bytes;
   uint32_t kind;
};

static struct radv_ps5_shader_note radv_ps5_shader_notes[RADV_PS5_SHADER_NOTES];
static uint32_t radv_ps5_shader_note_count;

void
radv_ps5_note_shader(uint64_t va, uint32_t bytes, const char *stage, uint32_t kind, uint64_t pipeline_hash,
                     uint64_t pipeline)
{
   if (!va || !bytes)
      return;
   const uint32_t slot = (p_atomic_inc_return(&radv_ps5_shader_note_count) - 1) % RADV_PS5_SHADER_NOTES;
   radv_ps5_shader_notes[slot] = (struct radv_ps5_shader_note){
      .va = va,
      .pipeline_hash = pipeline_hash,
      .pipeline = pipeline,
      .stage = stage ? stage : "?",
      .bytes = bytes,
      .kind = kind,
   };
}

/* Whose code is at address, newest first. Up to four are named: pipelines share a shader (RADV's
 * shader cache), and code freed with its pipeline leaves its entry behind for the next code put
 * at that address, so the newest entry is the likeliest. */
void eden_gpu_stalled(void) __attribute__((weak));

static void
radv_ps5_dump_shader_owner(const char *reg, uint64_t address)
{
   const uint32_t total = p_atomic_read(&radv_ps5_shader_note_count);
   const uint32_t kept = total < RADV_PS5_SHADER_NOTES ? total : RADV_PS5_SHADER_NOTES;
   uint32_t found = 0;
   for (uint32_t i = 0; i < kept && found < 4; i++) {
      const struct radv_ps5_shader_note *const note = &radv_ps5_shader_notes[(total - 1 - i) % RADV_PS5_SHADER_NOTES];
      if (address < note->va || address - note->va >= note->bytes)
         continue;
      const uint32_t kind = note->kind & ~RADV_PS5_SHADER_INTERNAL;
      if (kind == RADV_PS5_SHADER_PROLOG) {
         if (!found)
            fprintf(stderr,
                    "radv/ps5 stall: %s 0x%" PRIx64 " = a vertex-input prolog (RADV's own; the draw's vertex shader "
                    "runs after it, in the pipeline its PS names)\n",
                    reg, address);
         found++;
         break;
      }
      fprintf(stderr, "radv/ps5 stall: %s 0x%" PRIx64 " = pipeline %016" PRIx64 " vk=0x%" PRIx64 " (%s %s%s%s)\n", reg,
              address, note->pipeline_hash, note->pipeline, kind == RADV_PS5_SHADER_COMPUTE ? "compute" : "graphics",
              note->stage, (note->kind & RADV_PS5_SHADER_INTERNAL) ? ", RADV internal" : "",
              found ? ", recorded earlier at this address" : "");
      found++;
   }
   if (!found)
      fprintf(stderr, "radv/ps5 stall: %s 0x%" PRIx64 " = no recorded shader (%u recorded, the newest %u kept)\n", reg,
              address, total, kept);
}

/* ------------------------------------------------------- the stuck submission, in depth */

#define RADV_PS5_STALL_WAITS 256
#define RADV_PS5_STALL_WRITES 1024
#define RADV_PS5_STALL_WORK 16
#define RADV_PS5_STALL_WAIT_LINES 64

/* The program address registers the walk follows (GFX10.3: the ES registers hold the NGG or the
 * legacy GS stage's program, the LS registers the HS stage's), in pipeline order. */
enum {
   RADV_PS5_STAGE_VS,
   RADV_PS5_STAGE_HS,
   RADV_PS5_STAGE_GS,
   RADV_PS5_STAGE_PS,
   RADV_PS5_STAGE_CS,
   RADV_PS5_STAGES,
};
static const char *const radv_ps5_stage_names[RADV_PS5_STAGES] = {"VS", "HS", "GS", "PS", "CS"};

enum radv_ps5_wait_state {
   RADV_PS5_WAIT_MET,         /* its condition holds now */
   RADV_PS5_WAIT_PASSED,      /* it does not, but a later write in the submission left the value */
   RADV_PS5_WAIT_UNMET,       /* it does not, and nothing later in the submission explains why */
   RADV_PS5_WAIT_UNKNOWN,     /* the CPU cannot see what it waits on */
   RADV_PS5_WAIT_CONDITIONAL, /* a COND_EXEC: no wait, a run of dwords it may skip */
};

struct radv_ps5_stall_wait {
   uint32_t at;      /* the packet's first dword in the submission */
   uint32_t op;
   uint32_t number;  /* 1 for the submission's first wait; 0 for a COND_EXEC */
   uint32_t control; /* WAIT_REG_MEM's function and space; COND_EXEC's dword count */
   uint32_t ref;
   uint32_t mask;
   uint32_t value; /* what the CPU reads at address now, if readable */
   uint64_t address; /* memory, or a register's dword offset */
   bool memory;
   bool readable;
   uint8_t state;
};

struct radv_ps5_stall_write {
   uint32_t at;
   uint32_t value;
   uint64_t address;
};

struct radv_ps5_stall_work {
   uint32_t at;
   uint32_t op;
   uint32_t ordinal;
   uint64_t address[RADV_PS5_STAGES];
   /* How much work it asked for: a direct dispatch's groups in x, y and z; a direct draw's vertex or
    * index count, and its instances when a NUM_INSTANCES before it in this submission set them (0:
    * not known here). An indirect draw's are in memory: its first draw's count and instances as the
    * CPU reads them now (args_read), and for a multi draw how many draws it makes (draws, 0: one). */
   uint32_t size[3];
   uint32_t instances;
   bool args_read;
   bool args_unreadable;
   uint32_t draws;
};

/* The packet at words[i] (i < count): its opcode and its length in dwords (a type-2 filler is one
 * dword with no opcode). False at a word that starts no packet, or a packet that runs past count. */
static bool
radv_ps5_stall_packet(const uint32_t *words, uint32_t count, uint32_t i, uint32_t *op, uint32_t *n)
{
   const uint32_t header = words[i];
   if (PKT_TYPE_G(header) == 2) {
      *op = UINT32_MAX;
      *n = 1;
      return true;
   }
   if (PKT_TYPE_G(header) != 3)
      return false;
   *op = PKT3_IT_OPCODE_G(header);
   const uint32_t body = PKT_COUNT_G(header);
   *n = (*op == PKT3_NOP && body == 0x3fff) ? 1 : body + 2;
   return *n <= count - i;
}

/* WAIT_REG_MEM's FUNCTION field. */
static const char *const radv_ps5_wait_functions[8] = {"always", "<", "<=", "==", "!=", ">=", ">", "reserved"};

static bool
radv_ps5_stall_holds(uint32_t function, uint32_t value, uint32_t ref)
{
   switch (function) {
   case 0: return true;
   case 1: return value < ref;
   case 2: return value <= ref;
   case 3: return value == ref;
   case 4: return value != ref;
   case 5: return value >= ref;
   case 6: return value > ref;
   default: return false;
   }
}

/* Which program address registers a draw runs, by VGT_SHADER_STAGES_EN: every one when the walk
 * has not seen it. With NGG (PRIMGEN_EN) the VS stage does not run; the GS stage runs the vertex
 * or tessellation evaluation shader. */
static bool
radv_ps5_stall_stage_runs(unsigned stage, bool known, uint32_t stages)
{
   if (!known)
      return true;
   switch (stage) {
   case RADV_PS5_STAGE_VS: return !G_028B54_PRIMGEN_EN(stages);
   case RADV_PS5_STAGE_HS: return G_028B54_HS_EN(stages);
   case RADV_PS5_STAGE_GS: return G_028B54_ES_EN(stages) || G_028B54_GS_EN(stages) || G_028B54_PRIMGEN_EN(stages);
   default: return true;
   }
}

/* The program address register (byte offset) reg is, as a stage and whether it is the high half;
 * -1 for any other register. */
static int
radv_ps5_stall_program_reg(uint32_t reg, bool *high)
{
   *high = false;
   switch (reg) {
   case R_00B124_SPI_SHADER_PGM_HI_VS: *high = true; return RADV_PS5_STAGE_VS;
   case R_00B120_SPI_SHADER_PGM_LO_VS: return RADV_PS5_STAGE_VS;
   case R_00B524_SPI_SHADER_PGM_HI_LS: *high = true; return RADV_PS5_STAGE_HS;
   case R_00B520_SPI_SHADER_PGM_LO_LS: return RADV_PS5_STAGE_HS;
   case R_00B324_SPI_SHADER_PGM_HI_ES: *high = true; return RADV_PS5_STAGE_GS;
   case R_00B320_SPI_SHADER_PGM_LO_ES: return RADV_PS5_STAGE_GS;
   case R_00B024_SPI_SHADER_PGM_HI_PS: *high = true; return RADV_PS5_STAGE_PS;
   case R_00B020_SPI_SHADER_PGM_LO_PS: return RADV_PS5_STAGE_PS;
   case R_00B834_COMPUTE_PGM_HI: *high = true; return RADV_PS5_STAGE_CS;
   case R_00B830_COMPUTE_PGM_LO: return RADV_PS5_STAGE_CS;
   default: return -1;
   }
}

static void
radv_ps5_dump_wait_text(const struct radv_ps5_stall_wait *w)
{
   if (w->op == PKT3_WAIT_ON_CE_COUNTER) {
      fprintf(stderr, "WAIT_ON_CE_COUNTER");
   } else if (w->op == PKT3_COND_EXEC) {
      fprintf(stderr, "COND_EXEC mem 0x%" PRIx64 " (the next %u dwords run if it is not 0)", w->address, w->control);
   } else if (w->memory) {
      fprintf(stderr, "WAIT_REG_MEM mem 0x%" PRIx64 " & 0x%08x %s 0x%08x", w->address, w->mask,
              radv_ps5_wait_functions[w->control & 0x7], w->ref);
   } else {
      fprintf(stderr, "WAIT_REG_MEM reg 0x%05" PRIx64 " & 0x%08x %s 0x%08x", w->address * 4, w->mask,
              radv_ps5_wait_functions[w->control & 0x7], w->ref);
   }
   if (w->readable)
      fprintf(stderr, ", now 0x%08x", w->value);
}

static void
radv_ps5_dump_wait_state(const struct radv_ps5_stall_wait *w, bool reached, bool stop)
{
   if (!reached) {
      fprintf(stderr, "not reached\n");
   } else if (stop) {
      fprintf(stderr, "UNMET, the first: the GPU stops here\n");
   } else if (w->state == RADV_PS5_WAIT_CONDITIONAL) {
      if (!w->readable)
         fprintf(stderr, "cannot tell: the CPU cannot read that memory\n");
      else
         fprintf(stderr, "%s\n", w->value ? "runs" : "skips");
   } else if (w->state == RADV_PS5_WAIT_MET) {
      fprintf(stderr, "met\n");
   } else if (w->state == RADV_PS5_WAIT_PASSED) {
      fprintf(stderr, "passed (a later write in this submission left that value)\n");
   } else if (w->op == PKT3_WAIT_ON_CE_COUNTER) {
      fprintf(stderr, "cannot tell: the constant engine's counter\n");
   } else if (!w->memory) {
      fprintf(stderr, "cannot tell: a register\n");
   } else {
      fprintf(stderr, "cannot tell: the CPU cannot read that memory\n");
   }
}

/* What the stall dump shows of the last emulated task draw (radv_cmd_buffer.c, radv_task_emu_begin):
 * [0] the task emulation buffer, [1] the publish ring (0 when the mesh shader does not publish),
 * [2] ring entries, [3] payload stride, [4] mesh prim_parts | publishes << 8 | mesh has_query << 16 |
 * task has_query << 24, [5] occlusion | pipeline-stat << 16 | prims-gen << 32 queries active,
 * [6] task draws recorded, [7] mesh shader address. */
uint64_t radv_ps5_task_emu_note[8];

static void
radv_ps5_dump_words(const char *what, uint64_t va, uint32_t dwords)
{
   for (uint32_t k = 0; k < dwords; k += 8) {
      fprintf(stderr, "radv/ps5 stall: %s +0x%03x:", what, k * 4);
      for (uint32_t j = k; j < k + 8 && j < dwords; j++) {
         uint32_t v;
         if (radv_ps5_stall_read(va + 4 * j, &v))
            fprintf(stderr, " %08x", v);
         else
            fprintf(stderr, " ????????");
      }
      fprintf(stderr, "\n");
   }
}

static void
radv_ps5_dump_task_emu(void)
{
   const uint64_t *const t = radv_ps5_task_emu_note;
   if (!t[0])
      return;
   fprintf(stderr,
           "radv/ps5 stall: task emulation: bo 0x%" PRIx64 " publish ring 0x%" PRIx64 " entries %" PRIu64
           " stride %" PRIu64 " mesh parts %u publishes %u mesh query %u task query %u; queries occlusion %u"
           " stats %u prims-gen %u; %" PRIu64 " task draws; mesh 0x%" PRIx64 "\n",
           t[0], t[1], t[2], t[3], (unsigned)(t[4] & 0xff), (unsigned)((t[4] >> 8) & 0xff),
           (unsigned)((t[4] >> 16) & 0xff), (unsigned)((t[4] >> 24) & 0xff), (unsigned)(t[5] & 0xffff),
           (unsigned)((t[5] >> 16) & 0xffff), (unsigned)(t[5] >> 32), t[6], t[7]);
   radv_ps5_dump_words("records", t[0] /* RADV_TASK_EMU_RECORDS_OFFSET */, 2 * 16);
   radv_ps5_dump_words("draw ring", t[0] + 65536u * 64u /* RADV_TASK_EMU_DRAW_RING_OFFSET */, 4 * 4);
   radv_ps5_dump_words("payload", t[0] + 65536u * 64u + 65536u * 16u /* RADV_TASK_EMU_PAYLOAD_OFFSET */,
                       t[3] ? (uint32_t)MIN2(t[3] / 4 * 2, 64) : 16);
   if (t[1])
      radv_ps5_dump_words("publish headers", t[1], 16);
   fflush(stderr);
}

/* Every packet from the last wait the GPU passed to a little past where it stopped: opcode, length
 * and the first body dwords. */
static void
radv_ps5_dump_packets(const uint32_t *words, uint32_t count, uint32_t from, uint32_t to)
{
   uint32_t at = 0, op, n, shown = 0;
   fprintf(stderr, "radv/ps5 stall: packets from dword %u to %u:\n", from, to);
   for (; at < count && at <= to && radv_ps5_stall_packet(words, count, at, &op, &n); at += n) {
      if (at < from || op == UINT32_MAX)
         continue;
      if (++shown > 400) {
         fprintf(stderr, "radv/ps5 stall:   ...\n");
         break;
      }
      const char *name = radv_ps5_pkt3_name(op);
      fprintf(stderr, "radv/ps5 stall:   @%u op 0x%02x %s n=%u:", at, op, name ? name : "", n);
      for (uint32_t k = 1; k < n && k <= 8; k++)
         fprintf(stderr, " %08x", words[at + k]);
      fprintf(stderr, "%s\n", n > 9 ? " ..." : "");
   }
   fflush(stderr);
}

/* The stuck submission, packet by packet: the waits it carries and what each reads now, the first
 * whose condition does not hold (where the in-order GPU stopped), the last draws and dispatches
 * before it with the program each runs, and whose code that is. */
static void
radv_ps5_dump_stuck(const uint32_t *words, uint32_t count, uint64_t seq)
{
   static struct radv_ps5_stall_wait waits[RADV_PS5_STALL_WAITS];
   static struct radv_ps5_stall_write writes[RADV_PS5_STALL_WRITES];
   static struct radv_ps5_stall_work draws[RADV_PS5_STALL_WORK], dispatches[RADV_PS5_STALL_WORK];
   uint32_t wait_count = 0, wait_total = 0, cond_total = 0, write_count = 0;
   uint32_t at = 0, op, n;

   /* 1. The waits, the conditional runs and the memory writes (WRITE_DATA and RELEASE_MEM: what
    *    explains a counter a wait has been passed on), in order. */
   for (; at < count && radv_ps5_stall_packet(words, count, at, &op, &n); at += n) {
      const uint32_t *const body = words + at + 1;
      const uint32_t body_dw = n - 1;
      if (op == PKT3_WAIT_REG_MEM || op == PKT3_WAIT_ON_CE_COUNTER || op == PKT3_COND_EXEC) {
         const uint32_t number = op == PKT3_COND_EXEC ? 0 : ++wait_total;
         cond_total += op == PKT3_COND_EXEC;
         if (wait_count == RADV_PS5_STALL_WAITS)
            continue;
         struct radv_ps5_stall_wait *const w = &waits[wait_count];
         memset(w, 0, sizeof(*w));
         w->at = at;
         w->op = op;
         w->number = number;
         if (op == PKT3_WAIT_REG_MEM && body_dw >= 5) {
            w->control = body[0];
            w->memory = ((body[0] >> 4) & 0x3) == 1;
            w->address = w->memory ? (((uint64_t)body[2] << 32) | (body[1] & ~3u)) : body[1];
            w->ref = body[3];
            w->mask = body[4];
         } else if (op == PKT3_COND_EXEC && body_dw >= 4) {
            w->memory = true;
            w->address = ((uint64_t)body[1] << 32) | (body[0] & ~3u);
            w->control = body[3] & 0x3fff;
         } else if (op != PKT3_WAIT_ON_CE_COUNTER) {
            continue; /* too short to be what it says */
         }
         wait_count++;
      } else if (op == PKT3_WRITE_DATA && body_dw >= 4) {
         /* DST_SEL: memory (1 synchronised across GRBM, 5), or L2 (2). */
         const uint32_t dst = (body[0] >> 8) & 0xf;
         const uint64_t address = ((uint64_t)body[2] << 32) | (body[1] & ~3u);
         for (uint32_t k = 3; (dst == 1 || dst == 2 || dst == 5) && k < body_dw && k < 7 &&
                              write_count < RADV_PS5_STALL_WRITES;
              k++)
            writes[write_count++] =
               (struct radv_ps5_stall_write){.at = at, .value = body[k], .address = address + 4 * (k - 3)};
      } else if (op == PKT3_RELEASE_MEM && body_dw >= 6) {
         /* DST_SEL (17:16) memory or L2; DATA_SEL (31:29) the 32- or 64-bit value. */
         const uint32_t dst = (body[1] >> 16) & 0x3, data = body[1] >> 29;
         const uint64_t address = ((uint64_t)body[3] << 32) | (body[2] & ~3u);
         for (uint32_t k = 0; dst <= 1 && (data == 1 || data == 2) && k < data && write_count < RADV_PS5_STALL_WRITES;
              k++)
            writes[write_count++] =
               (struct radv_ps5_stall_write){.at = at, .value = body[4 + k], .address = address + 4 * k};
      }
   }
   const uint32_t parsed = at;

   /* 2. What each reads now, and the first wait whose condition does not hold. A wait on a counter
    *    the submission counts up does not hold once the counter has moved past it either; a later
    *    write of the value it reads now says the GPU got past it. */
   int stop_wait = -1;
   uint32_t unknown = 0;
   for (uint32_t k = 0; k < wait_count; k++) {
      struct radv_ps5_stall_wait *const w = &waits[k];
      if (w->memory)
         w->readable = radv_ps5_stall_read(w->address, &w->value);
      if (w->op == PKT3_COND_EXEC) {
         w->state = RADV_PS5_WAIT_CONDITIONAL;
      } else if (!w->readable) {
         w->state = RADV_PS5_WAIT_UNKNOWN;
         unknown++;
      } else if (radv_ps5_stall_holds(w->control & 0x7, w->value & w->mask, w->ref)) {
         w->state = RADV_PS5_WAIT_MET;
      } else {
         w->state = RADV_PS5_WAIT_UNMET;
         for (uint32_t j = 0; j < write_count; j++) {
            if (writes[j].at > w->at && writes[j].address == w->address && writes[j].value == w->value) {
               w->state = RADV_PS5_WAIT_PASSED;
               break;
            }
         }
         if (w->state == RADV_PS5_WAIT_UNMET && stop_wait < 0)
            stop_wait = (int)k;
      }
   }
   const uint32_t stop = stop_wait >= 0 ? waits[stop_wait].at : parsed;

   /* 3. The draws and dispatches before that point, with the program each runs as the SET_SH_REG
    *    packets before it left the program address registers. */
   uint32_t lo[RADV_PS5_STAGES] = {0}, hi[RADV_PS5_STAGES] = {0};
   uint32_t stages = 0, draw_total = 0, dispatch_total = 0, instances = 0;
   uint64_t indirect_base = 0; /* SET_BASE 1: where an indirect draw's data_offset points into */
   bool stages_known = false;
   for (at = 0; at < stop && radv_ps5_stall_packet(words, count, at, &op, &n); at += n) {
      const uint32_t *const body = words + at + 1;
      const uint32_t body_dw = n - 1;
      if ((op == PKT3_SET_SH_REG || op == PKT3_SET_SH_REG_INDEX) && body_dw >= 2) {
         const uint32_t reg = SI_SH_REG_OFFSET + (body[0] & 0xffff) * 4;
         for (uint32_t k = 1; k < body_dw; k++) {
            bool high;
            const int stage = radv_ps5_stall_program_reg(reg + (k - 1) * 4, &high);
            if (stage >= 0)
               (high ? hi : lo)[stage] = body[k];
         }
      } else if (op == PKT3_SET_CONTEXT_REG && body_dw >= 2) {
         const uint32_t reg = SI_CONTEXT_REG_OFFSET + (body[0] & 0xffff) * 4;
         if (R_028B54_VGT_SHADER_STAGES_EN >= reg && R_028B54_VGT_SHADER_STAGES_EN < reg + (body_dw - 1) * 4) {
            stages = body[1 + (R_028B54_VGT_SHADER_STAGES_EN - reg) / 4];
            stages_known = true;
         }
      } else if (op == PKT3_NUM_INSTANCES && body_dw >= 1) {
         instances = body[0];
      } else if (op == PKT3_SET_BASE && body_dw >= 3 && (body[0] & 0xf) == 1) {
         indirect_base = body[1] | ((uint64_t)body[2] << 32);
      } else if (radv_ps5_pkt3_is_draw(op) || op == PKT3_DISPATCH_DIRECT || op == PKT3_DISPATCH_INDIRECT) {
         const bool draw = radv_ps5_pkt3_is_draw(op);
         const uint32_t ordinal = draw ? ++draw_total : ++dispatch_total;
         struct radv_ps5_stall_work *const slot = &(draw ? draws : dispatches)[(ordinal - 1) % RADV_PS5_STALL_WORK];
         memset(slot, 0, sizeof(*slot));
         slot->at = at;
         slot->op = op;
         slot->ordinal = ordinal;
         if (op == PKT3_DISPATCH_DIRECT && body_dw >= 3) {
            slot->size[0] = body[0];
            slot->size[1] = body[1];
            slot->size[2] = body[2];
         } else if (draw) {
            /* DRAW_INDEX_AUTO: count, initiator. DRAW_INDEX_2: max, base lo, base hi, count, initiator.
             * DRAW_INDEX_OFFSET_2: max, offset, count, initiator. */
            if (op == PKT3_DRAW_INDEX_AUTO && body_dw >= 1)
               slot->size[0] = body[0];
            else if (op == PKT3_DRAW_INDEX_2 && body_dw >= 4)
               slot->size[0] = body[3];
            else if (op == PKT3_DRAW_INDEX_OFFSET_2 && body_dw >= 3)
               slot->size[0] = body[2];
            slot->instances = instances;
            /* DRAW_(INDEX_)INDIRECT(_MULTI): data_offset first; a multi draw then has its most draws
             * at body[4], and the address of its count at body[5..6] when COUNT_INDIRECT_ENABLE (bit
             * 30 of body[3]) is set. The first draw's arguments open with its count and instances. */
            const bool indirect = op == PKT3_DRAW_INDIRECT || op == PKT3_DRAW_INDEX_INDIRECT ||
                                  op == PKT3_DRAW_INDIRECT_MULTI || op == PKT3_DRAW_INDEX_INDIRECT_MULTI;
            if (indirect && indirect_base && body_dw >= 1) {
               const uint64_t args = indirect_base + body[0];
               slot->instances = 0;
               slot->args_read = radv_ps5_stall_read(args, &slot->size[0]) &&
                                 radv_ps5_stall_read(args + 4, &slot->instances);
               slot->args_unreadable = !slot->args_read;
               if (slot->args_read && (op == PKT3_DRAW_INDIRECT_MULTI || op == PKT3_DRAW_INDEX_INDIRECT_MULTI) &&
                   body_dw >= 7) {
                  slot->draws = body[4];
                  const uint64_t count_va = body[5] | ((uint64_t)body[6] << 32);
                  uint32_t draw_count;
                  if ((body[3] & (1u << 30)) && count_va && radv_ps5_stall_read(count_va, &draw_count) &&
                      draw_count < slot->draws)
                     slot->draws = draw_count;
               }
            }
         }
         for (unsigned s = 0; s < RADV_PS5_STAGES; s++) {
            const bool runs = draw ? s != RADV_PS5_STAGE_CS && radv_ps5_stall_stage_runs(s, stages_known, stages)
                                   : s == RADV_PS5_STAGE_CS;
            if (runs && lo[s])
               slot->address[s] = ((uint64_t)lo[s] << 8) | ((uint64_t)(hi[s] & 0xff) << 40);
         }
      }
   }

   /* What matters most goes first: the console may be gone before the rest is out. */
   if (stop_wait >= 0) {
      const struct radv_ps5_stall_wait *const w = &waits[stop_wait];
      fprintf(stderr, "radv/ps5 stall: seq %" PRIu64 " stops at wait %u of %u, dword %u: ", seq, w->number, wait_total,
              w->at);
      radv_ps5_dump_wait_text(w);
      fprintf(stderr, "\n");
   } else {
      fprintf(stderr,
              "radv/ps5 stall: seq %" PRIu64 ": none of its %u waits reads as unmet (%u the CPU cannot tell)%s\n",
              seq, wait_total, unknown, parsed < count ? ", and its words stop parsing early" : "");
   }
   fprintf(stderr, "radv/ps5 stall: the last draws and dispatches before dword %u (%u draws, %u dispatches before it):\n",
           stop, draw_total, dispatch_total);
   const struct radv_ps5_stall_work *work[2 * RADV_PS5_STALL_WORK];
   uint32_t work_count = 0;
   for (uint32_t k = 0; k < RADV_PS5_STALL_WORK; k++) {
      if (k < draw_total)
         work[work_count++] = &draws[k];
      if (k < dispatch_total)
         work[work_count++] = &dispatches[k];
   }
   for (uint32_t k = 1; k < work_count; k++) { /* in stream order */
      const struct radv_ps5_stall_work *const item = work[k];
      uint32_t j = k;
      for (; j > 0 && work[j - 1]->at > item->at; j--)
         work[j] = work[j - 1];
      work[j] = item;
   }
   const char *owner_reg[2 * RADV_PS5_STALL_WORK * RADV_PS5_STAGES];
   uint64_t owner_address[2 * RADV_PS5_STALL_WORK * RADV_PS5_STAGES];
   uint32_t owners = 0;
   for (uint32_t k = 0; k < work_count; k++) {
      const struct radv_ps5_stall_work *const item = work[k];
      const bool draw = radv_ps5_pkt3_is_draw(item->op);
      fprintf(stderr, "radv/ps5 stall:   %s %u at dword %u %s", draw ? "draw" : "dispatch", item->ordinal, item->at,
              radv_ps5_pkt3_name(item->op));
      if (!draw && item->op == PKT3_DISPATCH_DIRECT)
         fprintf(stderr, " %ux%ux%u groups", item->size[0], item->size[1], item->size[2]);
      else if (draw && item->args_read) {
         fprintf(stderr, " args count %u x%u instances", item->size[0], item->instances);
         if (item->draws > 1)
            fprintf(stderr, ", the first of %u draws", item->draws);
      } else if (draw && item->args_unreadable)
         fprintf(stderr, " args unreadable");
      else if (draw && item->size[0])
         fprintf(stderr, " count %u", item->size[0]);
      if (draw && !item->args_read && item->size[0] && item->instances)
         fprintf(stderr, " x%u instances", item->instances);
      fprintf(stderr, ":");
      bool any = false;
      for (unsigned s = 0; s < RADV_PS5_STAGES; s++) {
         if (!item->address[s])
            continue;
         any = true;
         fprintf(stderr, " %s 0x%" PRIx64, radv_ps5_stage_names[s], item->address[s]);
         bool seen = false;
         for (uint32_t j = 0; j < owners && !seen; j++)
            seen = owner_address[j] == item->address[s] && owner_reg[j] == radv_ps5_stage_names[s];
         if (!seen) {
            owner_reg[owners] = radv_ps5_stage_names[s];
            owner_address[owners++] = item->address[s];
         }
      }
      fprintf(stderr, "%s\n", any ? "" : " no program address set");
   }
   for (uint32_t k = 0; k < owners; k++)
      radv_ps5_dump_shader_owner(owner_reg[k], owner_address[k]);
   fflush(stderr);

   fprintf(stderr, "radv/ps5 stall: seq %" PRIu64 " carries %u waits and %u conditional runs:\n", seq, wait_total,
           cond_total);
   const uint32_t lines = wait_count < RADV_PS5_STALL_WAIT_LINES ? wait_count : RADV_PS5_STALL_WAIT_LINES;
   for (uint32_t k = 0; k < lines; k++) {
      const struct radv_ps5_stall_wait *const w = &waits[k];
      if (w->number)
         fprintf(stderr, "radv/ps5 stall:   wait %u at dword %u: ", w->number, w->at);
      else
         fprintf(stderr, "radv/ps5 stall:   cond at dword %u: ", w->at);
      radv_ps5_dump_wait_text(w);
      fprintf(stderr, ": ");
      radv_ps5_dump_wait_state(w, stop_wait < 0 || (int)k <= stop_wait, (int)k == stop_wait);
   }
   if (wait_total + cond_total > lines)
      fprintf(stderr, "radv/ps5 stall:   (%u more not shown)\n", wait_total + cond_total - lines);
   fflush(stderr);

   /* The waits around the stop, then every packet since the last wait the GPU passed. */
   uint32_t from = 0;
   if (stop_wait >= 0) {
      const uint32_t first = stop_wait > 3 ? (uint32_t)stop_wait - 3 : 0;
      for (uint32_t k = first; k < wait_count && k <= (uint32_t)stop_wait + 2; k++) {
         const struct radv_ps5_stall_wait *const w = &waits[k];
         fprintf(stderr, "radv/ps5 stall:   near the stop, %s %u at dword %u: ", w->number ? "wait" : "cond",
                 w->number, w->at);
         radv_ps5_dump_wait_text(w);
         fprintf(stderr, ": ");
         radv_ps5_dump_wait_state(w, (int)k <= stop_wait, (int)k == stop_wait);
      }
      for (int k = stop_wait - 1; k >= 0; k--) {
         if (waits[k].op == PKT3_WAIT_REG_MEM) {
            from = waits[k].at;
            break;
         }
      }
      if (stop - from > 6000)
         from = stop - 6000;
   } else {
      from = parsed > 2000 ? parsed - 2000 : 0;
   }
   radv_ps5_dump_packets(words, count, from, stop + 64);
   radv_ps5_dump_task_emu();
}

static void
radv_ps5_dump_stall(struct radv_ps5_queue *queue, uint64_t seq, uint64_t completed)
{
   /* Once per process: the console may hang right after, and a short burst is all it can flush. */
   static uint32_t dumped;
   if (p_atomic_cmpxchg(&dumped, 0u, 1u) != 0u)
      return;
   /* Before anything here can fault: the app learns its GPU stopped, so a crash in this dump
    * closes it as VK_ERROR_DEVICE_LOST does instead of starting it again on the stuck GPU
    * (headless/crash_report.cpp). Weak: a build without the app has no such hook. */
   if (eden_gpu_stalled)
      eden_gpu_stalled();
   /* A lock-free, best-effort snapshot: taking the submit lock could deadlock when the stall
    * trips inside radv_ps5_queue_claim, which already holds it, and at a ten-second hang the
    * submitting thread is quiescent anyway. Indices are masked, and radv_ps5_record_words
    * refuses a record whose words fall outside their buffer. */
   const uint32_t first = p_atomic_read(&queue->in_flight_first);
   const uint32_t total = p_atomic_read(&queue->in_flight_count);
   fprintf(stderr,
           "radv/ps5 stall: waiting for submission %" PRIu64 ", GPU at %" PRIu64 ", submitted %" PRIu64
           ", %u in flight (best-effort snapshot)\n",
           seq, completed, p_atomic_read(&queue->submitted_seq), total);
   const uint32_t window = total < 8 ? total : 8;
   for (uint32_t i = 0; i < window; i++)
      radv_ps5_dump_in_flight(queue, &queue->in_flight[(first + i) % RADV_PS5_MAX_IN_FLIGHT], completed);
   fflush(stderr);
   /* The stuck one in depth: the queue runs in order, so it is the one after the GPU's. */
   const uint32_t searched = total < RADV_PS5_MAX_IN_FLIGHT ? total : RADV_PS5_MAX_IN_FLIGHT;
   for (uint32_t i = 0; i < searched; i++) {
      const struct radv_ps5_in_flight *const record = &queue->in_flight[(first + i) % RADV_PS5_MAX_IN_FLIGHT];
      if (record->seq != completed + 1)
         continue;
      const uint32_t *words;
      uint32_t count;
      if (radv_ps5_record_words(queue, record, &words, &count))
         radv_ps5_dump_stuck(words, count, record->seq);
      break;
   }
}

void
radv_ps5_record_submission(struct radv_ps5_queue *queue, uint32_t words, uint32_t parts, uint32_t wait_count,
                           uint32_t signal_count, uint32_t queue_index, uint64_t depends)
{
   if (!queue->in_flight_count)
      return;
   struct radv_ps5_in_flight *const record =
      &queue->in_flight[(queue->in_flight_first + queue->in_flight_count - 1) % RADV_PS5_MAX_IN_FLIGHT];
   record->words = words;
   record->parts = (uint16_t)parts;
   record->wait_count = (uint16_t)wait_count;
   record->signal_count = (uint16_t)signal_count;
   record->queue_index = (uint8_t)queue_index;
   record->depends = depends;
}

bool
radv_ps5_queue_wait_seq(struct radv_ps5_queue *queue, uint64_t seq, uint64_t abs_timeout_ns)
{
   uint64_t completed = radv_ps5_queue_poll(queue);
   if (completed >= seq)
      return true;
   uint64_t now = os_time_get_nano();
   const uint64_t spin_until = now + RADV_PS5_SPIN_NS;
   uint64_t progress_at = now;
   uint64_t last = completed;
   for (;;) {
      completed = radv_ps5_queue_poll(queue);
      if (completed >= seq)
         return true;
      now = os_time_get_nano();
      if (now >= abs_timeout_ns)
         return false;
      if (completed != last) {
         last = completed;
         progress_at = now;
      } else if (now - progress_at > RADV_PS5_STALL_NS) {
         radv_ps5_dump_stall(queue, seq, completed); /* ProsperoEden: tools/patch-radv-submission-dump.py */
         fprintf(stderr, "radv/ps5: the GPU has not reached submission %" PRIu64 " (at %" PRIu64 ") in %u s\n", seq,
                 completed, (unsigned)(RADV_PS5_STALL_NS / 1000000000));
         return false;
      }
      if (now < spin_until)
         __builtin_ia32_pause();
      else
         radv_ps5_sleep_us(1000);
   }
}

/* EglantinePS5: the stall dump when the app's hang watch sees no frame for a while
 * (headless/crash_report.cpp, eden_radv_stall_dump). radv_ps5_queue_wait_seq's ten-second check
 * runs only while a thread polls a submission's marker; a hang where the submitting thread is
 * blocked in sceAgcDriverSubmitDcb and every other thread waits on a Vulkan semaphore never reaches
 * it, and said nothing. Dumps when the oldest submission still in flight has not completed, as
 * that check would; the dump itself runs once per process and is lock-free. Never creates the
 * queue. */
static void
radv_ps5_dump_stall_now(void)
{
   struct radv_ps5_queue *const queue = p_atomic_read(&radv_ps5_queue_instance);
   if (!queue)
      return;
   uint64_t completed = radv_ps5_queue_poll(queue);
   uint64_t submitted = p_atomic_read(&queue->submitted_seq);
   if (completed < submitted) {
      /* A GPU merely behind (compiling shaders, say) moves on within a second: the dump, which
       * runs once, is for one that has stopped. */
      os_time_sleep(1000000);
      const uint64_t later = radv_ps5_queue_poll(queue);
      if (later > completed) {
         fprintf(stderr, "radv/ps5: stall dump asked: the GPU is still running (submission %" PRIu64
                 " to %" PRIu64 " in a second); not dumping\n", completed, later);
         return;
      }
      submitted = p_atomic_read(&queue->submitted_seq);
   }
   if (completed >= submitted) {
      fprintf(stderr, "radv/ps5: stall dump asked: the GPU has run every submission (%" PRIu64 ")\n", completed);
      return;
   }
   fprintf(stderr, "radv/ps5: stall dump asked: the GPU has run submission %" PRIu64 " of %" PRIu64 "\n", completed,
           submitted);
   radv_ps5_dump_stall(queue, completed + 1, completed);
}

/* The app's hook for it: a pointer the app defines (weak here, so a program without it links). */
extern void (*eden_radv_stall_dump)(void) __attribute__((weak));

static void
radv_ps5_offer_stall_dump(void)
{
   if (&eden_radv_stall_dump)
      p_atomic_set(&eden_radv_stall_dump, radv_ps5_dump_stall_now);
}

/* ----------------------------------------------------------- the GPU itself */

/* The console's GPU as the winsys describes it to RADV, in the form the amdgpu
 * kernel reports a GPU (Mesa's recorded devices, src/amd/common/amdgpu_devices.c,
 * use the same form for AddrLib's tests), then completed by ac_gpu_info's own
 * derivations.
 *
 * Measured on my console (PS5_Vulkan):
 * - shaders compiled by ACO for GFX10.3 (CHIP_NAVI21) run: every psbc stage
 *   since M5 A, and compute in waves of 32;
 * - tiled images follow AddrLib's GFX10 swizzles with sixteen pipes, a 256-byte
 *   pipe interleave and no RB+ -- the Navi10 configuration -- texel for texel in
 *   both 64 KiB modes (docs/HARDWARE_FINDINGS.md, tools/mip-layout-oracle.cpp);
 * - the GPU timestamp counts at 100 MHz (V0-query, pid 143);
 * - a full-screen draw's occlusion count is a sixteenth of its samples, which
 *   is one counter per render backend if there are sixteen (to be proved by S6).
 *
 * Not yet measured, and marked where they are set: the shader engine and
 * compute-unit counts (the public PS5 figures: two shader engines, 36 active
 * compute units of 40) and the render-backend layout. Probe S3 measures them;
 * until then they only size scratch and a few hardware limits. */
#define RADV_PS5_EXTERNAL_REV 0x28 /* the NAVI21 range: GFX10.3 */
/* The revision of the GC 10.1.3 model (radv_ps5_take_gc_10_1_3_traits): the
 * start of GFX1013's range. */
#define RADV_PS5_GFX1013_REV 0x82
#define RADV_PS5_GB_ADDR_CONFIG 0x00100044

/* A GFX10 GPU with the console's units, clocks and memory, as the amdgpu
 * kernel would report it and ac_gpu_info completes it; the IP version and the
 * revision decide which GFX10 it is. */
static void
radv_ps5_fill_info(struct radeon_info *info, bool compiler_compat_mode, uint32_t ip_minor,
                   uint32_t ip_discovery_version, uint32_t external_rev)
{
   const struct drm_amdgpu_info_hw_ip gfx_ip = {
      .hw_ip_version_major = 10,
      .hw_ip_version_minor = ip_minor,
      .ib_start_alignment = 32,
      .ib_size_alignment = 32,
      .available_rings = 0x1,
      .ip_discovery_version = ip_discovery_version,
   };
   struct drm_amdgpu_info_device dev = {
      .device_id = 0,
      .chip_rev = 0,
      .external_rev = external_rev,
      .family = FAMILY_NV,
      /* Provisional until S3. */
      .num_shader_engines = 2,
      .num_shader_arrays_per_engine = 2,
      .cu_active_number = 36,
      .cu_ao_mask = 0x1ff,
      .cu_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .cu_ao_bitmap = {{0x1ff, 0x1ff}, {0x1ff, 0x1ff}},
      .num_cu_per_sh = 10,
      .enabled_rb_pipes_mask = 0xffff,
      .num_rb_pipes = 16,
      .num_tcc_blocks = 16,
      /* The 100 MHz timestamp clock, in kHz. */
      .gpu_counter_freq = 100000,
      .max_engine_clock = UINT64_C(2230000),
      .min_engine_clock = UINT64_C(500000),
      .max_memory_clock = UINT64_C(1750000),
      .min_memory_clock = UINT64_C(96000),
      .num_hw_gfx_contexts = 8,
      .ids_flags = 0,
      .virtual_address_offset = UINT64_C(0x200000),
      .virtual_address_max = RADV_PS5_GPU_ADDRESS_LIMIT,
      .virtual_address_alignment = RADV_PS5_PAGE_BYTES,
      .pte_fragment_size = 2097152,
      .gart_page_size = RADV_PS5_PAGE_BYTES,
      .vram_type = AMDGPU_VRAM_TYPE_GDDR6,
      .vram_bit_width = 256,
      .gc_double_offchip_lds_buf = 1,
      .wave_front_size = 32,
      .num_shader_visible_vgprs = 1024,
      .gs_vgt_table_depth = 32,
      .gs_prim_buffer_depth = 1792,
      .max_gs_waves_per_vgt = 32,
   };
   const uint64_t pool = radv_ps5_memory_pool_bytes();
   const struct drm_amdgpu_memory_info memory = {
      .vram = {.total_heap_size = pool},
      .cpu_accessible_vram = {.total_heap_size = pool},
      .gtt = {0},
   };
   const struct amdgpu_gpu_info gpu_info = {
      .gb_addr_cfg = RADV_PS5_GB_ADDR_CONFIG,
   };

   memset(info, 0, sizeof(*info));
   /* No kernel stands behind this description; the version is the lowest
    * RADV's amdgpu path accepts, so no feature is assumed from a newer kernel. */
   info->drm_major = 3;
   info->drm_minor = 54;
   info->is_amdgpu = false;

   ac_fill_hw_ip_info(info, &dev, AMD_IP_GFX, &gfx_ip);
   info->ip[AMD_IP_GFX].num_instances = 1;
   ac_identify_chip(info, &dev);
   ac_fill_memory_info(info, &dev, &memory);
   ac_fill_hw_info(info, &dev);
   ac_fill_tiling_info(info, &gpu_info);
   ac_fill_feature_info(info, &dev);
   ac_fill_bug_info(info);
   ac_fill_tess_info(info);
   ac_fill_compiler_info(info, &dev, compiler_compat_mode);
}

/* The GPU's fixed-function blocks are GC 10.1.3's, the chip of the AMD BC-250
 * (Mesa's CHIP_GFX1013, "GFX10 plus ray tracing instructions"; upstream amdgpu
 * programs it as GC 10.1.3 with the GB_ADDR_CONFIG above), while register
 * programming and the shader core behave as GFX10.3's, the identity RADV gets.
 * So these traits come from Mesa's GFX1013 model, derived from the same inputs.
 * Each was first found on the console as a symptom (PS5_Vulkan RADV_PHASE.md,
 * 2026-09-27, has the audit), and the model gives every one of them; the
 * assertion keeps the model and the measurements in step across Mesa updates.
 *
 * - No accelerated dot products: v_dot4_u32_u8 does not compute here. NGG's
 *   workgroup repack sums its counts with it, and a geometry shader whose
 *   vertex count is not constant, or a tessellated patch NGG culling repacked,
 *   drew nothing; with the v_msad_u8 fallback both draw every texel (the RADV
 *   smoke title's geometry and tessellation checks).
 * - No GFX10.3 wave ID in TG_SIZE: a 128-invocation workgroup of wave64
 *   reported one subgroup where two ran
 *   (dEQP-VK.subgroups.multiple_dispatches.uniform_subgroup_size); bits 20-24
 *   read 0, and the ordered wave ID serves instead.
 * - No per-primitive NGG parameters: a fragment shader reading gl_PrimitiveID
 *   that no earlier stage writes read 0 behind an NGG vertex shader
 *   (dEQP-VK.pipeline.*.misc.implicit_primitive_id; the smoke title's
 *   primitive ID check).
 * - No strict vertex order: the second of two triangles read its per-vertex
 *   inputs rotated with or without ROTATE_PC_PTR, so
 *   VK_KHR_fragment_shader_barycentric is not reported (the smoke title's
 *   barycentric pair; dEQP-VK.fragment_shading_barycentric.data).
 * - No variable-rate shading: a 2x2 pipeline rate ran the fragment shader once
 *   a texel and read gl_ShadingRateEXT as 1x1 (the smoke title's shading-rate
 *   check; dEQP-VK.fragment_shading_rate), so VK_KHR_fragment_shading_rate is
 *   not reported. No RB+ either.
 * - No E5B9G9R9 colour target: every blit into it with a non-zero colour read
 *   back wrong (dEQP-VK.api.copy_and_blit.*.blit_image.all_formats.color.*.
 *   e5b9g9r9_ufloat_pack32, 500 cases).
 * - The TC-compatible HTILE clear bug: a depth-only image cleared to 0 read back
 *   1 after a draw whose fragments were all discarded
 *   (dEQP-VK.dynamic_state.*.discard.depth); its workaround sets
 *   ZRANGE_PRECISION 0 after a clear to 0.
 * - GFX1013's AddrLib revision: non-RB+ GFX10 swizzles with the depth/stencil
 *   mipmap fix Navi10 lacks. With a Navi10 revision a 256x256 D16 image with
 *   mips was laid out in three 64 KiB blocks, its 128x128 level in the mip tail,
 *   while the depth block wrote level 0 into a fourth
 *   (dEQP-VK.glsl.texture_functions.texture.sampler2dshadow_*).
 *
 * GFX1013's model differs from GFX10.3 in more than these; the rest is not
 * taken until measured here (the audit lists them). */
static void
radv_ps5_take_gc_10_1_3_traits(struct radeon_info *info, bool compiler_compat_mode)
{
   struct radeon_info model;
   radv_ps5_fill_info(&model, compiler_compat_mode, 1, 0xa0103, RADV_PS5_GFX1013_REV);
   assert(model.family == CHIP_GFX1013);

   info->compiler_info.has_accelerated_dot_product = model.compiler_info.has_accelerated_dot_product;
   info->compiler_info.has_cs_wave_id = model.compiler_info.has_cs_wave_id;
   info->compiler_info.has_ngg_per_prim_params = model.compiler_info.has_ngg_per_prim_params;
   info->has_ps_strict_vertex_order = model.has_ps_strict_vertex_order;
   info->has_vrs = model.has_vrs;
   info->rbplus_allowed = model.rbplus_allowed;
   info->has_rgb9e5_color_target = model.has_rgb9e5_color_target;
   info->has_htile_tc_z_clear_bug_without_stencil = model.has_htile_tc_z_clear_bug_without_stencil;
   info->has_htile_tc_z_clear_bug_with_stencil = model.has_htile_tc_z_clear_bug_with_stencil;
   info->chip_external_rev = model.chip_external_rev;

   assert(!info->compiler_info.has_accelerated_dot_product && !info->compiler_info.has_cs_wave_id &&
          !info->compiler_info.has_ngg_per_prim_params && !info->has_ps_strict_vertex_order && !info->has_vrs &&
          !info->rbplus_allowed && !info->has_rgb9e5_color_target && info->has_htile_tc_z_clear_bug_without_stencil &&
          info->has_htile_tc_z_clear_bug_with_stencil && info->chip_external_rev == RADV_PS5_GFX1013_REV);
}

static void
radv_ps5_describe_gpu(struct radeon_info *info, bool compiler_compat_mode)
{
   radv_ps5_fill_info(info, compiler_compat_mode, 3, 0xa0300, RADV_PS5_EXTERNAL_REV);
   snprintf(info->marketing_name, sizeof(info->marketing_name), "PlayStation 5 GPU");
   radv_ps5_take_gc_10_1_3_traits(info, compiler_compat_mode);

   /* This GPU's parameter cache rotates per-vertex fragment inputs and tells the shader nothing about it: the compiler
    * restores the order itself (docs/BARYCENTRIC_GFX101.md), so VK_KHR_fragment_shader_barycentric is reported. */
   info->compiler_info.emulate_ps_strict_vertex_order = true;

   /* What the platform, not the chip, rules out. */
   /* Every wave of a shader using scratch_* instructions faulted (MEMVIOL) a
    * few instructions in, where it had set FLAT_SCRATCH with s_setreg; the
    * same shaders through buffer instructions, as on GFX8, run (the smoke
    * title's scratch checks, graphicsfuzz's large private arrays). */
   info->compiler_info.has_flat_scratch = false;
   /* The BVH instruction is GFX10.3's (and GC 10.1.3's): acceleration
    * structures build and ray queries traverse them (dEQP-VK.ray_query and
    * the ray query cases elsewhere, run rq-full-1 in PS5_Vulkan), so the
    * description keeps it. */

   /* A legacy GS hung the GPU every time (dEQP-VK.geometry with
    * RADV_DEBUG=nongg: 31 of 33 cases). Why is not established: RADV sets the
    * GS ring sizes with a UCONFIG write, as it does everywhere, while the
    * tessellation rings are AGC's (sceAgcDriverSetTFRing); whether the GS
    * ring sizes are too is unmeasured. */
   info->has_legacy_gs = false;
   /* An INDIRECT_BUFFER into title memory faulted the GPU (PS5_Vulkan B8), so
    * the command buffers device-generated commands write cannot run. */
   info->has_gpu_written_ibs = false;
   /* VideoOut scans out 64 KiB R_X tiles (tiling 0) with no pipe or bank
    * swizzle: ps5vk renders its framebuffers in that mode, and the demo
    * renderer's CPU-written tiles showed (PS5_Vulkan src/demo_renderer.cpp,
    * ps5platform/videoout.h). */
   info->has_fixed_display_swizzle = true;
   info->display_swizzle_mode = ADDR_SW_64KB_R_X;
   /* Every indirect mesh draw stopped the CP with a bad opcode at
    * DISPATCH_MESH_INDIRECT_MULTI (dEQP-VK.mesh_shader.ext.api.draw_indirect*,
    * run mesh-exp-1 in PS5_Vulkan); direct mesh draws, DRAW_INDEX_AUTO in
    * fast launch, run. Indirect ones go through ordinary indirect draws
    * (radv_mesh_draw_records_enabled). */
   info->has_dispatch_mesh_indirect = false;
   /* No exported function holds the GPU at a stable power state
    * (radv_ps5_ctx_set_pstate refuses every state but none), so the profiling
    * lock VK_KHR_performance_query takes always failed
    * (dEQP-VK.query_pool.performance_query.*: VK_ERROR_UNKNOWN). */
   info->has_perf_counters = false;
   /* One pool of memory the CPU and the GPU share, which RADV presents as an
    * integrated GPU's: a device-local heap and a host heap over it, with
    * host-cached types. The CPU's mapping of that memory is cached and the GPU
    * reads and writes it coherently with the CPU's caches (the RADV smoke
    * title's coherence probe). */
   info->has_dedicated_vram = false;
   info->all_vram_visible = true;
   info->address32_hi = RADV_PS5_ADDRESS32_HI;
   /* Timeline semaphores: the runtime builds them over the binary sync type
    * (vk_sync_timeline), with threaded submission for waits before signals. */
   info->has_timeline_syncobj = true;
   info->kernel_has_modifiers = false;
   /* Sparse resources are ranges of the device-memory region whose pages the
    * winsys maps to buffers as they are bound, and to a shared zero block
    * while they are not (radv_ps5_vrange_*). No exported function sets the
    * page table's PRT bits, so an unbound page is not known to read zero or
    * to report itself non-resident; and a bind is a mapping change made at
    * once, after the submission's waits. */
   info->has_sparse = true;
   /* A compute and transfer queue family, as upstream offers on this family
    * of GPUs, whose queues submit to the graphics ring through the same
    * sceAgcDriverSubmitDcb: their work runs in turn with graphics rather than
    * alongside it, which Vulkan allows. AGC's asynchronous compute queues
    * would run it alongside; they are not used yet. */
   info->num_compute_queues_on_gfx = 4;
   info->has_sparse_prt = false;
   info->has_sparse_cpu_binds = true;
   info->max_sparse_va_bytes = radv_ps5_vrange_space_bytes();
   /* A captured buffer can be placed at its address again
    * (radv_ps5_memory_alloc_replayable); before that, every replay case of
    * dEQP-VK.binding_model.buffer_device_address (723) failed. Outside the
    * shaders' window captures keep clear of every other buffer; in it the
    * kernel places buffers, so replayable ones (the ray tracing pipelines'
    * shader arenas, whose group handles are captured) go in a range at its
    * top the port keeps for them (radv_ps5_window_replayable). */
   info->has_replayable_va = true;
   info->has_replayable_va_32bit = radv_ps5_window_replayable();
   info->has_vm_always_valid = true;
   /* buffer_from_ptr gives the application's host memory GPU access, which the
    * GPU reaches at the CPU's address (HARDWARE_FINDINGS.md, 2026-09-27). */
   info->has_userptr = true;
   info->max_submitted_ibs[AMD_IP_GFX] = 1;

   /* ac_query_gpu_info's own derivations, which have no kernel input. */
   info->scratch_wavesize_granularity_shift = 10;
   info->scratch_wavesize_granularity = BITFIELD_BIT(info->scratch_wavesize_granularity_shift);
   const unsigned max_waves_per_tg = 32;
   info->max_scratch_waves =
      MAX2(32 * info->max_good_cu_per_sa * info->max_sa_per_se * info->num_se, max_waves_per_tg);
   info->has_scratch_base_registers = false;
   info->max_gflops = 128 * info->num_cu * info->max_gpu_freq_mhz / 1000;
   info->memory_bandwidth_gbps =
      DIV_ROUND_UP(info->memory_freq_mhz_effective * info->memory_bus_width / 8, 1000);
   info->instr_prefetch_distance = 3;
   info->se_tile_repeat = 32 * info->max_se;
}

void
radv_ps5_winsys_heap_usage(uint64_t *pool_bytes, uint64_t *available_bytes, uint64_t *driver_bytes)
{
   *pool_bytes = radv_ps5_memory_pool_bytes();
   *available_bytes = radv_ps5_memory_available_bytes();
   *driver_bytes = p_atomic_read(&radv_ps5_allocated_bytes);
}

VkResult
radv_ps5_winsys_query_info(uint64_t debug_flags, struct radeon_winsys_info *info)
{
   memset(info, 0, sizeof(*info));
   if (!radv_ps5_queue_get())
      return VK_ERROR_INITIALIZATION_FAILED;
   radv_ps5_describe_gpu(&info->base, !(debug_flags & RADV_DEBUG_NO_CACHE_COMPAT));
   info->syncobj_sync_type = radv_ps5_sync_type;
   /* One queue, which a title's submissions all reach: low and medium, the
    * priorities an application may always have, both run there. A priority is
    * a scheduling hint, and with one queue there is nothing to order. */
   info->global_priority_mask = BITFIELD_BIT(RADEON_CTX_PRIORITY_LOW) | BITFIELD_BIT(RADEON_CTX_PRIORITY_MEDIUM);
   /* GPU memory and the queue's syncs have no file descriptor to share. */
   info->has_external_fd = false;
   return VK_SUCCESS;
}

/* -------------------------------------------------------------- the winsys */

static uint64_t
radv_ps5_winsys_query_value(struct radeon_winsys *rws, enum radeon_value_id value)
{
   struct radv_ps5_winsys *ws = radv_ps5_winsys(rws);
   switch (value) {
   case RADEON_ALLOCATED_VRAM:
   case RADEON_ALLOCATED_VRAM_VIS:
   case RADEON_VRAM_USAGE:
   case RADEON_VRAM_VIS_USAGE:
      return p_atomic_read(&ws->allocated_vram);
   case RADEON_ALLOCATED_GTT:
   case RADEON_GTT_USAGE:
      return p_atomic_read(&ws->allocated_gtt);
   case RADEON_TIMESTAMP:
      return radv_ps5_read_gpu_clock();
   default:
      /* The console's other clocks and counters are not the title's to read. */
      return 0;
   }
}

static bool
radv_ps5_winsys_read_registers(struct radeon_winsys *rws, unsigned reg_offset, unsigned num_registers, uint32_t *out)
{
   (void)rws;
   (void)reg_offset;
   (void)num_registers;
   (void)out;
   return false;
}

static bool
radv_ps5_winsys_query_gpuvm_fault(struct radeon_winsys *rws, struct radv_winsys_gpuvm_fault_info *fault_info)
{
   (void)rws;
   (void)fault_info;
   return false;
}

static VkResult
radv_ps5_ctx_create(struct radeon_winsys *rws, enum radeon_ctx_priority priority, struct radeon_winsys_ctx **rctx)
{
   /* The priorities reported (global_priority_mask) are the ones a context is
    * made with; a higher one is denied, as the extension allows
    * (dEQP-VK.api.device_init.create_device_global_priority*). */
   if (priority != RADEON_CTX_PRIORITY_MEDIUM && priority != RADEON_CTX_PRIORITY_LOW)
      return VK_ERROR_NOT_PERMITTED;
   struct radv_ps5_ctx *const ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   ctx->ws = radv_ps5_winsys(rws);
   ctx->priority = priority;
   *rctx = (struct radeon_winsys_ctx *)ctx;
   return VK_SUCCESS;
}

static void
radv_ps5_ctx_set_tess_factor_ring(struct radeon_winsys_ctx *rctx, uint64_t va, uint32_t size)
{
   struct radv_ps5_ctx *const ctx = (struct radv_ps5_ctx *)rctx;
   ctx->tess_factor_ring_va = va;
   ctx->tess_factor_ring_size = size;
}

static void
radv_ps5_ctx_destroy(struct radeon_winsys_ctx *rctx)
{
   free(rctx);
}

static bool
radv_ps5_ctx_wait_idle(struct radeon_winsys_ctx *rctx, enum amd_ip_type ip_type, int ring_index)
{
   (void)ip_type;
   (void)ring_index;
   struct radv_ps5_queue *const queue = ((struct radv_ps5_ctx *)rctx)->ws->queue;
   return radv_ps5_queue_wait_seq(queue, p_atomic_read(&queue->submitted_seq), OS_TIMEOUT_INFINITE);
}

static int
radv_ps5_ctx_set_pstate(struct radeon_winsys_ctx *rctx, uint32_t pstate)
{
   (void)rctx;
   return pstate == RADEON_CTX_PSTATE_NONE ? 0 : -1;
}

static void
radv_ps5_winsys_destroy(struct radeon_winsys *rws)
{
   free(rws);
}

static int
radv_ps5_winsys_get_fd(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static struct util_sync_provider *
radv_ps5_winsys_get_sync_provider(struct radeon_winsys *rws)
{
   (void)rws;
   return NULL;
}

static int
radv_ps5_winsys_reserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
   return -1;
}

static void
radv_ps5_winsys_unreserve_vmid(struct radeon_winsys *rws)
{
   (void)rws;
}

VkResult
radv_ps5_winsys_create(const struct radeon_info *info, uint64_t debug_flags, uint64_t perftest_flags,
                       struct radeon_winsys **winsys)
{
   struct radv_ps5_queue *const queue = radv_ps5_queue_get();
   if (!queue)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct radv_ps5_winsys *const ws = calloc(1, sizeof(*ws));
   if (!ws)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy(&ws->info, info, sizeof(ws->info));
   ws->debug_flags = debug_flags;
   ws->perftest_flags = perftest_flags;
   ws->queue = queue;

   ws->base.destroy = radv_ps5_winsys_destroy;
   ws->base.query_value = radv_ps5_winsys_query_value;
   ws->base.read_registers = radv_ps5_winsys_read_registers;
   ws->base.query_gpuvm_fault = radv_ps5_winsys_query_gpuvm_fault;
   ws->base.ctx_create = radv_ps5_ctx_create;
   ws->base.ctx_destroy = radv_ps5_ctx_destroy;
   ws->base.ctx_set_tess_factor_ring = radv_ps5_ctx_set_tess_factor_ring;
   ws->base.ctx_wait_idle = radv_ps5_ctx_wait_idle;
   ws->base.ctx_set_pstate = radv_ps5_ctx_set_pstate;
   ws->base.get_fd = radv_ps5_winsys_get_fd;
   ws->base.get_sync_provider = radv_ps5_winsys_get_sync_provider;
   ws->base.copy_sync_payloads = radv_ps5_sync_copy_payloads;
   ws->base.reserve_vmid = radv_ps5_winsys_reserve_vmid;
   ws->base.unreserve_vmid = radv_ps5_winsys_unreserve_vmid;
   radv_ps5_bo_init_functions(ws);
   radv_ps5_cs_init_functions(ws);

   *winsys = &ws->base;
   return VK_SUCCESS;
}
