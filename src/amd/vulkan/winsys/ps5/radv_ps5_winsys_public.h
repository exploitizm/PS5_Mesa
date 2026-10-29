/*
 * Copyright © 2026 Mihawk
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_PS5_WINSYS_PUBLIC_H
#define RADV_PS5_WINSYS_PUBLIC_H

#include "radv_radeon_winsys.h"
#include "ac_gpu_info.h"
#include "vk_sync.h"

/* What the physical device learns from the winsys before it exists. */
struct radeon_winsys_info {
   struct radeon_info base;
   struct vk_sync_type syncobj_sync_type;
   uint32_t global_priority_mask;
   /* Memory, semaphores and fences can be shared as file descriptors
    * (opaque fd, sync fd, dma-buf). */
   bool has_external_fd;
};

/* The console's GPU: a GFX10.3 shader core addressed the way the console's
 * memory was measured to be tiled. Fails when AGC cannot be initialised. */
VkResult radv_ps5_winsys_query_info(uint64_t debug_flags, struct radeon_winsys_info *info);

VkResult radv_ps5_winsys_create(const struct radeon_info *info, uint64_t debug_flags, uint64_t perftest_flags,
                                struct radeon_winsys **winsys);

/* The direct-memory pool, what it has left, and what this driver holds of it,
 * for VK_EXT_memory_budget. */
void radv_ps5_winsys_heap_usage(uint64_t *pool_bytes, uint64_t *available_bytes, uint64_t *driver_bytes);

/* ProsperoEden (tools/patch-radv-submission-dump.py): a shader's code sits at [va, va + bytes) on
 * the GPU, for the stall dump to say whose code a stuck draw or dispatch was running: the stage, the
 * kind of code (RADV_PS5_SHADER_*), the pipeline's hash and its VkPipeline handle (0 for code no
 * pipeline owns). stage is a string that lives as long as the process. Called at pipeline and
 * vertex-input prolog creation: a few stores into a fixed ring, no allocation, no lock. */
#define RADV_PS5_SHADER_GRAPHICS 0u
#define RADV_PS5_SHADER_COMPUTE 1u
#define RADV_PS5_SHADER_PROLOG 2u
#define RADV_PS5_SHADER_INTERNAL 0x100u /* RADV's own: a meta pipeline or a prolog */
void radv_ps5_note_shader(uint64_t va, uint32_t bytes, const char *stage, uint32_t kind, uint64_t pipeline_hash,
                          uint64_t pipeline);

#endif /* RADV_PS5_WINSYS_PUBLIC_H */
