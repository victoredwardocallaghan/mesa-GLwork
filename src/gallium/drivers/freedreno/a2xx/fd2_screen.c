/* -*- mode: C; c-file-style: "k&r"; tab-width 4; indent-tabs-mode: t; -*- */

/*
 * Copyright (C) 2013 Rob Clark <robclark@freedesktop.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "pipe/p_screen.h"
#include "util/u_format.h"

#include "fd2_screen.h"
#include "fd2_context.h"
#include "fd2_util.h"

static boolean
fd2_screen_is_format_supported(struct pipe_screen *pscreen,
		enum pipe_format format,
		enum pipe_texture_target target,
		unsigned sample_count,
		unsigned usage)
{
	unsigned retval = 0;

	if ((target >= PIPE_MAX_TEXTURE_TYPES) ||
			(sample_count > 1) || /* TODO add MSAA */
			!util_format_is_supported(format, usage)) {
		DBG("not supported: format=%s, target=%d, sample_count=%d, usage=%x",
				util_format_name(format), target, sample_count, usage);
		return FALSE;
	}

	/* TODO figure out how to render to other formats.. */
	if ((usage & PIPE_BIND_RENDER_TARGET) &&
			((format != PIPE_FORMAT_B8G8R8A8_UNORM) &&
			 (format != PIPE_FORMAT_B8G8R8X8_UNORM))) {
		DBG("not supported render target: format=%s, target=%d, sample_count=%d, usage=%x",
				util_format_name(format), target, sample_count, usage);
		return FALSE;
	}

	if ((usage & (PIPE_BIND_SAMPLER_VIEW |
				PIPE_BIND_VERTEX_BUFFER)) &&
			(fd2_pipe2surface(format) != ~0u)) {
		retval |= usage & (PIPE_BIND_SAMPLER_VIEW |
				PIPE_BIND_VERTEX_BUFFER);
	}

	if ((usage & (PIPE_BIND_RENDER_TARGET |
				PIPE_BIND_DISPLAY_TARGET |
				PIPE_BIND_SCANOUT |
				PIPE_BIND_SHARED)) &&
			(fd2_pipe2color(format) != ~0u)) {
		retval |= usage & (PIPE_BIND_RENDER_TARGET |
				PIPE_BIND_DISPLAY_TARGET |
				PIPE_BIND_SCANOUT |
				PIPE_BIND_SHARED);
	}

	if ((usage & PIPE_BIND_DEPTH_STENCIL) &&
			(fd_pipe2depth(format) != ~0u)) {
		retval |= PIPE_BIND_DEPTH_STENCIL;
	}

	if ((usage & PIPE_BIND_INDEX_BUFFER) &&
			(fd_pipe2index(format) != ~0u)) {
		retval |= PIPE_BIND_INDEX_BUFFER;
	}

	if (usage & PIPE_BIND_TRANSFER_READ)
		retval |= PIPE_BIND_TRANSFER_READ;
	if (usage & PIPE_BIND_TRANSFER_WRITE)
		retval |= PIPE_BIND_TRANSFER_WRITE;

	if (retval != usage) {
		DBG("not supported: format=%s, target=%d, sample_count=%d, "
				"usage=%x, retval=%x", util_format_name(format),
				target, sample_count, usage, retval);
	}

	return retval == usage;
}

void
fd2_screen_init(struct pipe_screen *pscreen)
{
	fd_screen(pscreen)->max_rts = 1;
	pscreen->context_create = fd2_context_create;
	pscreen->is_format_supported = fd2_screen_is_format_supported;
}
