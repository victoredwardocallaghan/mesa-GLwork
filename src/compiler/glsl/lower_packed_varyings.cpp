/*
 * Copyright © 2011 Intel Corporation
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file lower_varyings_to_packed.cpp
 *
 * This lowering pass generates GLSL code that manually packs varyings into
 * vec4 slots, for the benefit of back-ends that don't support packed varyings
 * natively.
 *
 * For example, the following shader:
 *
 *   out mat3x2 foo;  // location=4, location_frac=0
 *   out vec3 bar[2]; // location=5, location_frac=2
 *
 *   main()
 *   {
 *     ...
 *   }
 *
 * Is rewritten to:
 *
 *   mat3x2 foo;
 *   vec3 bar[2];
 *   out vec4 packed4; // location=4, location_frac=0
 *   out vec4 packed5; // location=5, location_frac=0
 *   out vec4 packed6; // location=6, location_frac=0
 *
 *   main()
 *   {
 *     ...
 *     packed4.xy = foo[0];
 *     packed4.zw = foo[1];
 *     packed5.xy = foo[2];
 *     packed5.zw = bar[0].xy;
 *     packed6.x = bar[0].z;
 *     packed6.yzw = bar[1];
 *   }
 *
 * This lowering pass properly handles "double parking" of a varying vector
 * across two varying slots.  For example, in the code above, two of the
 * components of bar[0] are stored in packed5, and the remaining component is
 * stored in packed6.
 *
 * Note that in theory, the extra instructions may cause some loss of
 * performance.  However, hopefully in most cases the performance loss will
 * either be absorbed by a later optimization pass, or it will be offset by
 * memory bandwidth savings (because fewer varyings are used).
 *
 * This lowering pass also packs flat floats, ints, and uints together, by
 * using ivec4 as the base type of flat "varyings", and using appropriate
 * casts to convert floats and uints into ints.
 *
 * This lowering pass also handles varyings whose type is a struct or an array
 * of struct.  Structs are packed in order and with no gaps, so there may be a
 * performance penalty due to structure elements being double-parked.
 *
 * Lowering of geometry shader inputs is slightly more complex, since geometry
 * inputs are always arrays, so we need to lower arrays to arrays.  For
 * example, the following input:
 *
 *   in struct Foo {
 *     float f;
 *     vec3 v;
 *     vec2 a[2];
 *   } arr[3];         // location=4, location_frac=0
 *
 * Would get lowered like this if it occurred in a fragment shader:
 *
 *   struct Foo {
 *     float f;
 *     vec3 v;
 *     vec2 a[2];
 *   } arr[3];
 *   in vec4 packed4;  // location=4, location_frac=0
 *   in vec4 packed5;  // location=5, location_frac=0
 *   in vec4 packed6;  // location=6, location_frac=0
 *   in vec4 packed7;  // location=7, location_frac=0
 *   in vec4 packed8;  // location=8, location_frac=0
 *   in vec4 packed9;  // location=9, location_frac=0
 *
 *   main()
 *   {
 *     arr[0].f = packed4.x;
 *     arr[0].v = packed4.yzw;
 *     arr[0].a[0] = packed5.xy;
 *     arr[0].a[1] = packed5.zw;
 *     arr[1].f = packed6.x;
 *     arr[1].v = packed6.yzw;
 *     arr[1].a[0] = packed7.xy;
 *     arr[1].a[1] = packed7.zw;
 *     arr[2].f = packed8.x;
 *     arr[2].v = packed8.yzw;
 *     arr[2].a[0] = packed9.xy;
 *     arr[2].a[1] = packed9.zw;
 *     ...
 *   }
 *
 * But it would get lowered like this if it occurred in a geometry shader:
 *
 *   struct Foo {
 *     float f;
 *     vec3 v;
 *     vec2 a[2];
 *   } arr[3];
 *   in vec4 packed4[3];  // location=4, location_frac=0
 *   in vec4 packed5[3];  // location=5, location_frac=0
 *
 *   main()
 *   {
 *     arr[0].f = packed4[0].x;
 *     arr[0].v = packed4[0].yzw;
 *     arr[0].a[0] = packed5[0].xy;
 *     arr[0].a[1] = packed5[0].zw;
 *     arr[1].f = packed4[1].x;
 *     arr[1].v = packed4[1].yzw;
 *     arr[1].a[0] = packed5[1].xy;
 *     arr[1].a[1] = packed5[1].zw;
 *     arr[2].f = packed4[2].x;
 *     arr[2].v = packed4[2].yzw;
 *     arr[2].a[0] = packed5[2].xy;
 *     arr[2].a[1] = packed5[2].zw;
 *     ...
 *   }
 */

#include "glsl_symbol_table.h"
#include "ir.h"
#include "ir_builder.h"
#include "ir_optimization.h"
#include "ir_rvalue_visitor.h"
#include "linker.h"
#include "program/prog_instruction.h"
#include "util/hash_table.h"

using namespace ir_builder;

/**
 * If the var is an array check if it matches the array attributes of the
 * packed var.
 */
static bool
check_for_matching_arrays(ir_variable *packed_var, ir_variable *var)
{
   const glsl_type *pt = packed_var->type;
   const glsl_type *vt = var->type;
   bool array_match = true;

   while (pt->is_array() || vt->is_array()) {
      if (pt->is_array() != vt->is_array() ||
          pt->length != vt->length) {
         array_match = false;
         break;
      } else {
         pt = pt->fields.array;
         vt = vt->fields.array;
      }
   }

   return array_match;
}

/**
 * Creates new type for and array when the base type changes.
 */
static const glsl_type *
update_packed_array_type(const glsl_type *type, const glsl_type *packed_type)
{
   const glsl_type *element_type = type->fields.array;
   if (element_type->is_array()) {
     const glsl_type *new_array_type =
        update_packed_array_type(element_type, packed_type);
      return glsl_type::get_array_instance(new_array_type, type->length);
   } else {
      return glsl_type::get_array_instance(packed_type, type->length);
   }
}

static bool
needs_lowering(ir_variable *var, bool has_enhanced_layouts,
               bool disable_varying_packing)
{
   /* Don't lower varying with explicit location unless ARB_enhanced_layouts
    * is enabled, also don't try to pack structs with explicit location as
    * they don't support the component layout qualifier anyway.
    */
   if (var->data.explicit_location && (!has_enhanced_layouts ||
       var->type->without_array()->is_record())) {
      return false;
   }

   /* Don't disable packing for explicit locations when ARB_enhanced_layouts
    * is supported.
    */
   if (disable_varying_packing && !var->data.explicit_location)
      return false;

   /* Things composed of vec4's don't need lowering everything else does. */
   const glsl_type *type = var->type->without_array();
   if (type->vector_elements == 4 && !type->is_double())
      return false;
   return true;
}

static ir_variable *
create_packed_var(void * const mem_ctx, const char *packed_name,
                  const glsl_type *packed_type, ir_variable *unpacked_var,
                  ir_variable_mode mode, unsigned location,
                  bool is_outer_array_vert_idx)
{
   ir_variable *packed_var = new(mem_ctx)
      ir_variable(packed_type, packed_name, mode);
   if (is_outer_array_vert_idx) {
      /* Prevent update_array_sizes() from messing with the size of the
       * array.
       */
      packed_var->data.max_array_access = unpacked_var->type->length - 1;
   }
   packed_var->data.centroid = unpacked_var->data.centroid;
   packed_var->data.sample = unpacked_var->data.sample;
   packed_var->data.patch = unpacked_var->data.patch;
   packed_var->data.interpolation = unpacked_var->data.interpolation;
   packed_var->data.location = location;
   packed_var->data.precision = unpacked_var->data.precision;
   packed_var->data.always_active_io = unpacked_var->data.always_active_io;

   return packed_var;
}

/**
 * Creates a packed varying for the tessellation packing.
 */
static ir_variable *
create_tess_packed_var(void *mem_ctx, ir_variable *unpacked_var)
{
   /* create packed varying name using location */
   char location_str[11];
   snprintf(location_str, 11, "%d", unpacked_var->data.location);
   char *packed_name;
   if ((ir_variable_mode) unpacked_var->data.mode == ir_var_shader_out)
      packed_name = ralloc_asprintf(mem_ctx, "packed_out:%s", location_str);
   else
      packed_name = ralloc_asprintf(mem_ctx, "packed_in:%s", location_str);

   const glsl_type *packed_type;
   switch (unpacked_var->type->without_array()->base_type) {
   case GLSL_TYPE_UINT:
      packed_type = glsl_type::uvec4_type;
      break;
   case GLSL_TYPE_INT:
      packed_type = glsl_type::ivec4_type;
      break;
   case GLSL_TYPE_FLOAT:
      packed_type = glsl_type::vec4_type;
      break;
   case GLSL_TYPE_DOUBLE:
      packed_type = glsl_type::dvec4_type;
      break;
   default:
      assert(!"Unexpected type in tess varying packing");
      return NULL;
   }

   /* Create new array type */
   if (unpacked_var->type->is_array()) {
      packed_type = update_packed_array_type(unpacked_var->type, packed_type);
   }

   return create_packed_var(mem_ctx, packed_name, packed_type, unpacked_var,
                            (ir_variable_mode) unpacked_var->data.mode,
                            unpacked_var->data.location,
                            unpacked_var->type->is_array());
}

namespace {

/**
 * Visitor that performs varying packing.  For each varying declared in the
 * shader, this visitor determines whether it needs to be packed.  If so, it
 * demotes it to an ordinary global, creates new packed varyings, and
 * generates assignments to convert between the original varying and the
 * packed varying.
 */
class lower_packed_varyings_visitor
{
public:
   lower_packed_varyings_visitor(void *mem_ctx, unsigned locations_used,
                                 ir_variable_mode mode,
                                 bool is_outer_array_vert_idx,
                                 exec_list *out_instructions,
                                 exec_list *out_variables,
                                 unsigned base_location,
                                 bool disable_varying_packing,
                                 bool xfb_enabled,
                                 bool has_enhanced_layouts);

   void run(struct gl_shader *shader);

private:
   void bitwise_assign_pack(ir_rvalue *lhs, ir_rvalue *rhs);
   void bitwise_assign_unpack(ir_rvalue *lhs, ir_rvalue *rhs);
   unsigned lower_rvalue(ir_rvalue *rvalue, unsigned fine_location,
                         ir_variable *unpacked_var, const char *name,
                         bool is_outer_array_vert_idx, unsigned vertex_index,
                         bool explicit_location);
   unsigned lower_arraylike(ir_rvalue *rvalue, unsigned array_size,
                            unsigned fine_location,
                            ir_variable *unpacked_var, const char *name,
                            bool is_outer_array_vert_idx,
                            unsigned vertex_index, bool explicit_location);
   ir_dereference *get_packed_varying_deref(unsigned location,
                                            ir_variable *unpacked_var,
                                            const char *name,
                                            unsigned vertex_index);

   /**
    * Memory context used to allocate new instructions for the shader.
    */
   void * const mem_ctx;

   const unsigned base_location;

   /**
    * Number of generic varying slots which are used by this shader.  This is
    * used to allocate temporary intermediate data structures.  If any varying
    * used by this shader has a location greater than or equal to
    * base_location + locations_used, an assertion will fire.
    */
   const unsigned locations_used;

   /**
    * Array of pointers to the packed varyings that have been created for each
    * generic varying slot.  NULL entries in this array indicate varying slots
    * for which a packed varying has not been created yet.
    */
   ir_variable **packed_varyings;

   /**
    * Type of varying which is being lowered in this pass (either
    * ir_var_shader_in or ir_var_shader_out).
    */
   const ir_variable_mode mode;

   /**
    * Are we are currently lowering a stage where the input or output vertices
    * are indexed by the outmost array.
    */
   const bool is_outer_array_vert_idx;

   /**
    * Exec list into which the visitor should insert the packing instructions.
    * Caller provides this list; it should insert the instructions into the
    * appropriate place in the shader once the visitor has finished running.
    */
   exec_list *out_instructions;

   /**
    * Exec list into which the visitor should insert any new variables.
    */
   exec_list *out_variables;

   bool disable_varying_packing;
   bool xfb_enabled;
   bool has_enhanced_layouts;
};

} /* anonymous namespace */

lower_packed_varyings_visitor::lower_packed_varyings_visitor(
      void *mem_ctx, unsigned locations_used, ir_variable_mode mode,
      bool is_outer_array_vert_idx, exec_list *out_instructions,
      exec_list *out_variables, unsigned base_location,
      bool disable_varying_packing, bool xfb_enabled,
      bool has_enhanced_layouts)
   : mem_ctx(mem_ctx),
     base_location(base_location),
     locations_used(locations_used),
     packed_varyings((ir_variable **)
                     rzalloc_array_size(mem_ctx, sizeof(*packed_varyings),
                                        locations_used)),
     mode(mode),
     is_outer_array_vert_idx(is_outer_array_vert_idx),
     out_instructions(out_instructions),
     out_variables(out_variables),
     disable_varying_packing(disable_varying_packing),
     xfb_enabled(xfb_enabled),
     has_enhanced_layouts(has_enhanced_layouts)
{
}

void
lower_packed_varyings_visitor::run(struct gl_shader *shader)
{
   foreach_in_list(ir_instruction, node, shader->ir) {
      ir_variable *var = node->as_variable();
      if (var == NULL)
         continue;

      if (var->data.mode != this->mode ||
          var->data.location < (int) this->base_location ||
          !needs_lowering(var, has_enhanced_layouts, disable_varying_packing))
         continue;

      /* This lowering pass is only capable of packing floats and ints
       * together when their interpolation mode is "flat".  Therefore, to be
       * safe, caller should ensure that integral varyings always use flat
       * interpolation, even when this is not required by GLSL.
       */
      assert(var->data.interpolation == INTERP_QUALIFIER_FLAT ||
             !var->type->contains_integer());

      /* Clone the variable for program resource list before
       * it gets modified and lost.
       */
      if (!shader->packed_varyings)
         shader->packed_varyings = new (shader) exec_list;

      shader->packed_varyings->push_tail(var->clone(shader, NULL));

      /* Change the old varying into an ordinary global. */
      assert(var->data.mode != ir_var_temporary);
      var->data.mode = ir_var_auto;

      /* Create a reference to the old varying. */
      ir_dereference_variable *deref
         = new(this->mem_ctx) ir_dereference_variable(var);

      /* Recursively pack or unpack it. */
      this->lower_rvalue(deref, var->data.location * 4 + var->data.location_frac, var,
                         var->name, is_outer_array_vert_idx, 0,
                         var->data.explicit_location);
   }
}

#define SWIZZLE_ZWZW MAKE_SWIZZLE4(SWIZZLE_Z, SWIZZLE_W, SWIZZLE_Z, SWIZZLE_W)

/**
 * Make an ir_assignment from \c rhs to \c lhs, performing appropriate
 * bitcasts if necessary to match up types.
 *
 * This function is called when packing varyings.
 */
void
lower_packed_varyings_visitor::bitwise_assign_pack(ir_rvalue *lhs,
                                                   ir_rvalue *rhs)
{
   if (lhs->type->base_type != rhs->type->base_type) {
      /* Since we only mix types in flat varyings, and we always store flat
       * varyings as type ivec4, we need only produce conversions from (uint
       * or float) to int.
       */
      assert(lhs->type->base_type == GLSL_TYPE_INT);
      switch (rhs->type->base_type) {
      case GLSL_TYPE_UINT:
         rhs = new(this->mem_ctx)
            ir_expression(ir_unop_u2i, lhs->type, rhs);
         break;
      case GLSL_TYPE_FLOAT:
         rhs = new(this->mem_ctx)
            ir_expression(ir_unop_bitcast_f2i, lhs->type, rhs);
         break;
      case GLSL_TYPE_DOUBLE:
         assert(rhs->type->vector_elements <= 2);
         if (rhs->type->vector_elements == 2) {
            ir_variable *t = new(mem_ctx) ir_variable(lhs->type, "pack", ir_var_temporary);

            assert(lhs->type->vector_elements == 4);
            this->out_variables->push_tail(t);
            this->out_instructions->push_tail(
                  assign(t, u2i(expr(ir_unop_unpack_double_2x32, swizzle_x(rhs->clone(mem_ctx, NULL)))), 0x3));
            this->out_instructions->push_tail(
                  assign(t,  u2i(expr(ir_unop_unpack_double_2x32, swizzle_y(rhs))), 0xc));
            rhs = deref(t).val;
         } else {
            rhs = u2i(expr(ir_unop_unpack_double_2x32, rhs));
         }
         break;
      default:
         assert(!"Unexpected type conversion while lowering varyings");
         break;
      }
   }
   this->out_instructions->push_tail(new (this->mem_ctx) ir_assignment(lhs, rhs));
}


/**
 * Make an ir_assignment from \c rhs to \c lhs, performing appropriate
 * bitcasts if necessary to match up types.
 *
 * This function is called when unpacking varyings.
 */
void
lower_packed_varyings_visitor::bitwise_assign_unpack(ir_rvalue *lhs,
                                                     ir_rvalue *rhs)
{
   if (lhs->type->base_type != rhs->type->base_type) {
      /* Since we only mix types in flat varyings, and we always store flat
       * varyings as type ivec4, we need only produce conversions from int to
       * (uint or float).
       */
      assert(rhs->type->base_type == GLSL_TYPE_INT);
      switch (lhs->type->base_type) {
      case GLSL_TYPE_UINT:
         rhs = new(this->mem_ctx)
            ir_expression(ir_unop_i2u, lhs->type, rhs);
         break;
      case GLSL_TYPE_FLOAT:
         rhs = new(this->mem_ctx)
            ir_expression(ir_unop_bitcast_i2f, lhs->type, rhs);
         break;
      case GLSL_TYPE_DOUBLE:
         assert(lhs->type->vector_elements <= 2);
         if (lhs->type->vector_elements == 2) {
            ir_variable *t = new(mem_ctx) ir_variable(lhs->type, "unpack", ir_var_temporary);
            assert(rhs->type->vector_elements == 4);
            this->out_variables->push_tail(t);
            this->out_instructions->push_tail(
                  assign(t, expr(ir_unop_pack_double_2x32, i2u(swizzle_xy(rhs->clone(mem_ctx, NULL)))), 0x1));
            this->out_instructions->push_tail(
                  assign(t, expr(ir_unop_pack_double_2x32, i2u(swizzle(rhs->clone(mem_ctx, NULL), SWIZZLE_ZWZW, 2))), 0x2));
            rhs = deref(t).val;
         } else {
            rhs = expr(ir_unop_pack_double_2x32, i2u(rhs));
         }
         break;
      default:
         assert(!"Unexpected type conversion while lowering varyings");
         break;
      }
   }
   this->out_instructions->push_tail(new(this->mem_ctx) ir_assignment(lhs, rhs));
}


/**
 * Recursively pack or unpack the given varying (or portion of a varying) by
 * traversing all of its constituent vectors.
 *
 * \param fine_location is the location where the first constituent vector
 * should be packed--the word "fine" indicates that this location is expressed
 * in multiples of a float, rather than multiples of a vec4 as is used
 * elsewhere in Mesa.
 *
 * \param is_outer_array_vert_idx should be set to true if we are lowering an
 * array whose index selects a vertex e.g the outermost array of a geometry
 * shader input.
 *
 * \param vertex_index: if we are lowering geometry shader inputs, and the
 * level of the array that we are currently lowering is *not* the top level,
 * then this indicates which vertex we are currently lowering.  Otherwise it
 * is ignored.
 *
 * \return the location where the next constituent vector (after this one)
 * should be packed.
 */
unsigned
lower_packed_varyings_visitor::lower_rvalue(ir_rvalue *rvalue,
                                            unsigned fine_location,
                                            ir_variable *unpacked_var,
                                            const char *name,
                                            bool is_outer_array_vert_idx,
                                            unsigned vertex_index,
                                            bool explicit_location)
{
   unsigned dmul = rvalue->type->is_double() ? 2 : 1;
   /* When is_outer_array_vert_idx is set, we should be looking at a varying
    * array.
    */
   assert(!is_outer_array_vert_idx || rvalue->type->is_array());

   if (rvalue->type->is_record()) {
      for (unsigned i = 0; i < rvalue->type->length; i++) {
         if (i != 0)
            rvalue = rvalue->clone(this->mem_ctx, NULL);
         const char *field_name = rvalue->type->fields.structure[i].name;
         ir_dereference_record *dereference_record = new(this->mem_ctx)
            ir_dereference_record(rvalue, field_name);
         char *deref_name
            = ralloc_asprintf(this->mem_ctx, "%s.%s", name, field_name);
         fine_location = this->lower_rvalue(dereference_record, fine_location,
                                            unpacked_var, deref_name, false,
                                            vertex_index, explicit_location);
      }
      return fine_location;
   } else if (rvalue->type->is_array()) {
      /* Arrays are packed/unpacked by considering each array element in
       * sequence.
       */
      return this->lower_arraylike(rvalue, rvalue->type->array_size(),
                                   fine_location, unpacked_var, name,
                                   is_outer_array_vert_idx, vertex_index,
                                   explicit_location);
   } else if (rvalue->type->is_matrix()) {
      /* Matrices are packed/unpacked by considering each column vector in
       * sequence.
       */
      return this->lower_arraylike(rvalue, rvalue->type->matrix_columns,
                                   fine_location, unpacked_var, name,
                                   false, vertex_index, explicit_location);
   } else if (rvalue->type->vector_elements * dmul +
              fine_location % 4 > 4) {
      /* This vector is going to be "double parked" across two varying slots,
       * so handle it as two separate assignments. For doubles, a dvec3/dvec4
       * can end up being spread over 3 slots. However the second splitting
       * will happen later, here we just always want to split into 2.
       */
      unsigned left_components, right_components;
      unsigned left_swizzle_values[4] = { 0, 0, 0, 0 };
      unsigned right_swizzle_values[4] = { 0, 0, 0, 0 };
      char left_swizzle_name[4] = { 0, 0, 0, 0 };
      char right_swizzle_name[4] = { 0, 0, 0, 0 };

      left_components = 4 - fine_location % 4;
      if (rvalue->type->is_double()) {
         /* We might actually end up with 0 left components! */
         left_components /= 2;
      }
      right_components = rvalue->type->vector_elements - left_components;

      for (unsigned i = 0; i < left_components; i++) {
         left_swizzle_values[i] = i;
         left_swizzle_name[i] = "xyzw"[i];
      }
      for (unsigned i = 0; i < right_components; i++) {
         right_swizzle_values[i] = i + left_components;
         right_swizzle_name[i] = "xyzw"[i + left_components];
      }
      ir_swizzle *left_swizzle = new(this->mem_ctx)
         ir_swizzle(rvalue, left_swizzle_values, left_components);
      ir_swizzle *right_swizzle = new(this->mem_ctx)
         ir_swizzle(rvalue->clone(this->mem_ctx, NULL), right_swizzle_values,
                    right_components);
      char *left_name
         = ralloc_asprintf(this->mem_ctx, "%s.%s", name, left_swizzle_name);
      char *right_name
         = ralloc_asprintf(this->mem_ctx, "%s.%s", name, right_swizzle_name);
      if (left_components)
         fine_location = this->lower_rvalue(left_swizzle, fine_location,
                                            unpacked_var, left_name, false,
                                            vertex_index, explicit_location);
      else
         /* Top up the fine location to the next slot */
         fine_location++;
      return this->lower_rvalue(right_swizzle, fine_location, unpacked_var,
                                right_name, false, vertex_index,
                                explicit_location);
   } else {
      /* No special handling is necessary; pack the rvalue into the
       * varying.
       */
      unsigned swizzle_values[4] = { 0, 0, 0, 0 };
      unsigned components = rvalue->type->vector_elements * dmul;
      unsigned location = fine_location / 4;
      unsigned location_frac = fine_location % 4;
      for (unsigned i = 0; i < components; ++i)
         swizzle_values[i] = i + location_frac;
      ir_dereference *packed_deref =
         this->get_packed_varying_deref(location, unpacked_var, name,
                                        vertex_index);
      ir_swizzle *swizzle = new(this->mem_ctx)
         ir_swizzle(packed_deref, swizzle_values, components);
      if (this->mode == ir_var_shader_out) {
         this->bitwise_assign_pack(swizzle, rvalue);
      } else {
         this->bitwise_assign_unpack(rvalue, swizzle);
      }

      /* Explicitly packed components are packed by interleaving arrays, so
       * simply bump the location by 4 to increment the location to the next
       * element.
       *
       * Otherwise we pack arrays elements end to end.
       */
      if (explicit_location) {
         return fine_location + 4;
      } else
         return fine_location + components;
   }
}

/**
 * Recursively pack or unpack a varying for which we need to iterate over its
 * constituent elements, accessing each one using an ir_dereference_array.
 * This takes care of both arrays and matrices, since ir_dereference_array
 * treats a matrix like an array of its column vectors.
 *
 * \param is_outer_array_vert_idx should be set to true if we are lowering an
 * array whose index selects a vertex e.g the outermost array of a geometry
 * shader input.
 *
 * \param vertex_index: if we are lowering geometry shader inputs, and the
 * level of the array that we are currently lowering is *not* the top level,
 * then this indicates which vertex we are currently lowering.  Otherwise it
 * is ignored.
 */
unsigned
lower_packed_varyings_visitor::lower_arraylike(ir_rvalue *rvalue,
                                               unsigned array_size,
                                               unsigned fine_location,
                                               ir_variable *unpacked_var,
                                               const char *name,
                                               bool is_outer_array_vert_idx,
                                               unsigned vertex_index,
                                               bool explicit_location)
{
   for (unsigned i = 0; i < array_size; i++) {
      if (i != 0)
         rvalue = rvalue->clone(this->mem_ctx, NULL);
      ir_constant *constant = new(this->mem_ctx) ir_constant(i);
      ir_dereference_array *dereference_array = new(this->mem_ctx)
         ir_dereference_array(rvalue, constant);
      if (is_outer_array_vert_idx) {
         /* Geometry shader inputs are a special case.  Instead of storing
          * each element of the array at a different location, all elements
          * are at the same location, but with a different vertex index.
          */
         (void) this->lower_rvalue(dereference_array, fine_location,
                                   unpacked_var, name, false, i,
                                   explicit_location);
      } else {
         char *subscripted_name
            = ralloc_asprintf(this->mem_ctx, "%s[%d]", name, i);
         fine_location =
            this->lower_rvalue(dereference_array, fine_location,
                               unpacked_var, subscripted_name,
                               false, vertex_index, explicit_location);
      }
   }
   return fine_location;
}

/**
 * Retrieve the packed varying corresponding to the given varying location.
 * If no packed varying has been created for the given varying location yet,
 * create it and add it to the shader before returning it.
 *
 * The newly created varying inherits its interpolation parameters from \c
 * unpacked_var.  Its base type is ivec4 if we are lowering a flat varying,
 * vec4 otherwise.
 *
 * \param vertex_index: if we are lowering geometry shader inputs, then this
 * indicates which vertex we are currently lowering.  Otherwise it is ignored.
 */
ir_dereference *
lower_packed_varyings_visitor::get_packed_varying_deref(
      unsigned location, ir_variable *unpacked_var, const char *name,
      unsigned vertex_index)
{
   unsigned slot = location - this->base_location;
   assert(slot < locations_used);
   if (this->packed_varyings[slot] == NULL) {
      char *packed_name = ralloc_asprintf(this->mem_ctx, "packed:%s", name);
      const glsl_type *packed_type;
      if (unpacked_var->data.interpolation == INTERP_QUALIFIER_FLAT)
         packed_type = glsl_type::ivec4_type;
      else
         packed_type = glsl_type::vec4_type;
      if (this->is_outer_array_vert_idx) {
         packed_type =
            glsl_type::get_array_instance(packed_type,
                                          unpacked_var->type->length);
      }

      ir_variable *packed_var =
         create_packed_var(mem_ctx, packed_name, packed_type, unpacked_var,
                           this->mode, location,
                           this->is_outer_array_vert_idx);
      unpacked_var->insert_before(packed_var);
      this->packed_varyings[slot] = packed_var;
   } else {
      /* For geometry shader inputs, only update the packed variable name the
       * first time we visit each component.
       */
      if (!this->is_outer_array_vert_idx || vertex_index == 0) {
         ralloc_asprintf_append((char **) &this->packed_varyings[slot]->name,
                                ",%s", name);
      }
   }

   ir_dereference *deref = new(this->mem_ctx)
      ir_dereference_variable(this->packed_varyings[slot]);
   if (this->is_outer_array_vert_idx) {
      /* When lowering GS inputs, the packed variable is an array, so we need
       * to dereference it using vertex_index.
       */
      ir_constant *constant = new(this->mem_ctx) ir_constant(vertex_index);
      deref = new(this->mem_ctx) ir_dereference_array(deref, constant);
   }
   return deref;
}

/**
 * Visitor that splices varying packing code before every use of EmitVertex()
 * in a geometry shader.
 */
class lower_packed_varyings_gs_splicer : public ir_hierarchical_visitor
{
public:
   explicit lower_packed_varyings_gs_splicer(void *mem_ctx,
                                             const exec_list *instructions);

   virtual ir_visitor_status visit_leave(ir_emit_vertex *ev);

private:
   /**
    * Memory context used to allocate new instructions for the shader.
    */
   void * const mem_ctx;

   /**
    * Instructions that should be spliced into place before each EmitVertex()
    * call.
    */
   const exec_list *instructions;
};


lower_packed_varyings_gs_splicer::lower_packed_varyings_gs_splicer(
      void *mem_ctx, const exec_list *instructions)
   : mem_ctx(mem_ctx), instructions(instructions)
{
}


ir_visitor_status
lower_packed_varyings_gs_splicer::visit_leave(ir_emit_vertex *ev)
{
   foreach_in_list(ir_instruction, ir, this->instructions) {
      ev->insert_before(ir->clone(this->mem_ctx, NULL));
   }
   return visit_continue;
}


/**
 * For tessellation control shaders, we cannot just copy everything to the
 * packed varyings like we do in other stages.  TCS outputs can be used as
 * shared memory, where multiple threads concurrently perform partial reads
 * and writes that must not conflict.  It is only safe to access the exact
 * components that the shader uses.
 *
 * This class searches the IR for uses of varyings and then emits a copy for
 * everything it finds hoping later optimizations are able to clean up any
 * duplicates.
 */
class lower_packed_varyings_tess_visitor : public ir_rvalue_visitor
{
public:
   lower_packed_varyings_tess_visitor(void *mem_ctx, hash_table *varyings,
                                      ir_variable_mode mode)
   : mem_ctx(mem_ctx), varyings(varyings), mode(mode)
   {
   }

   virtual ~lower_packed_varyings_tess_visitor()
   {
   }

   virtual ir_visitor_status visit_leave(ir_assignment *);
   virtual ir_visitor_status visit_leave(ir_dereference_array *);

   ir_dereference *create_dereference(ir_dereference *deref,
                                      unsigned *dimensions,
                                      bool *has_vec_subscript);
   unsigned create_extra_array_dereference(unsigned inner_dimension,
                                           const glsl_type **types_list,
                                           ir_dereference **packed_deref_list,
                                           ir_dereference **deref_list);
   ir_variable *get_packed_var(ir_variable *var);
   void handle_rvalue(ir_rvalue **rvalue);

   /**
    * Exec list into which the visitor should insert the packing instructions.
    * Caller provides this list; it should insert the instructions into the
    * appropriate place in the shader once the visitor has finished running.
    */
   exec_list new_instructions;

private:
   /**
    * Memory context used to allocate new instructions for the shader.
    */
   void * const mem_ctx;

   hash_table *varyings;

   ir_variable_mode mode;
};

/**
 * Search the hash table for a packed varying for this variable.
 */
ir_variable *
lower_packed_varyings_tess_visitor::get_packed_var(ir_variable *var)
{
   assert(var);

   const struct hash_entry *entry =
      _mesa_hash_table_search(varyings, var);

   return entry ? (ir_variable *) entry->data : NULL;
}

ir_dereference *
lower_packed_varyings_tess_visitor::create_dereference(ir_dereference *deref,
                                                       unsigned *dimension,
                                                       bool *has_vec_subscript)
{
   ir_dereference_array *deref_array = deref->as_dereference_array();
   if (deref_array) {
      ir_dereference *array =
         create_dereference(deref_array->array->as_dereference(), dimension,
                            has_vec_subscript);

      /* The array dereference may actually be to access vector components
       * so don't touch the dimension count unless we are actually dealing
       * with an array.
       */
      if (deref_array->array->type->is_array()) {
         (*dimension)--;
      } else {
         /* If we have found a vector not an array don't create an array
          * dereference and set the has_vec_subscript flag so we can remove
          * the array dereference from the unpacked var too.
          */
         *has_vec_subscript = true;
         return array;
      }

      return new(this->mem_ctx)
         ir_dereference_array(array,
                              deref_array->array_index->clone(mem_ctx, NULL));
   } else {
      ir_variable *unpacked_var = deref->variable_referenced();
      ir_variable *packed_var = get_packed_var(unpacked_var);
      return new(this->mem_ctx) ir_dereference_variable(packed_var);
   }
}
/**
 * This creates the extra derefs needed to copy the full array. For example if
 * we have:
 *
 *    layout(location = 0, component = 3) in float b[][6];
 *    layout(location = 0, component = 3) out float b_tcs[][6];
 *    ...
 *    b_tcs[gl_InvocationID] = b[gl_InvocationID];
 *
 * We need to copy all the inner array elements to the new packed varying:
 *
 *    packed_out:26[gl_InvocationID][0].w = b_tcs[gl_InvocationID][0];
 *    ...
 *    packed_out:26[gl_InvocationID][5].w = b_tcs[gl_InvocationID][5];
 */
unsigned
lower_packed_varyings_tess_visitor::create_extra_array_dereference(unsigned inner_dimension,
                                                                   const glsl_type **types_list,
                                                                   ir_dereference **packed_deref_list,
                                                                   ir_dereference **deref_list)
{
   unsigned outer_deref_array_size;
   if (inner_dimension != 0)
      outer_deref_array_size =
         create_extra_array_dereference(inner_dimension - 1, types_list,
                                        packed_deref_list, deref_list);
    else {
      assert(types_list[inner_dimension]->length > 0);
      outer_deref_array_size = 1;
   }

   unsigned deref_array_size =
      types_list[inner_dimension]->length * outer_deref_array_size;

   /* Create new lists to store the new instructions in */
   ir_dereference **new_packed_deref_list = (ir_dereference **)
      rzalloc_array_size(mem_ctx, sizeof(ir_dereference *), deref_array_size);
   ir_dereference **new_deref_list = (ir_dereference **)
      rzalloc_array_size(mem_ctx, sizeof(ir_dereference *), deref_array_size);

   unsigned list_count = 0;
   for (unsigned i = 0; i < types_list[inner_dimension]->length; i++) {
      for (unsigned j = 0; j < outer_deref_array_size; j++) {
         /* Clone the outer dimension derefs */
         ir_dereference *deref_clone = deref_list[j]->clone(this->mem_ctx, NULL);
         ir_dereference *packed_deref_clone = packed_deref_list[j]->clone(this->mem_ctx, NULL);

         /* Create new derefs for the inner dimiension */
         ir_constant *constant = new(this->mem_ctx) ir_constant(i);
         new_packed_deref_list[list_count] = new(this->mem_ctx)
            ir_dereference_array(packed_deref_clone, constant);

         ir_constant *constant2 = new(this->mem_ctx) ir_constant(i);
         new_deref_list[list_count] = new(this->mem_ctx)
            ir_dereference_array(deref_clone, constant2);
         list_count++;
      }
   }

   /* Copy the new derefs so the caller can access them */
   for (unsigned j = 0; j < list_count; j++) {
      packed_deref_list[j] = new_packed_deref_list[j];
      deref_list[j] = new_deref_list[j];
   }
   return deref_array_size;
}

void
lower_packed_varyings_tess_visitor::handle_rvalue(ir_rvalue **rvalue)
{
   if (!*rvalue)
      return;

   ir_dereference *deref = (*rvalue)->as_dereference();

   if (!deref)
      return;

   ir_variable *unpacked_var = deref->variable_referenced();
   ir_variable *packed_var = get_packed_var(unpacked_var);

   /* If the variable is packed then create a new dereference and wrap it in
    * a swizzle to get the correct values as specified by the component
    * qualifier.
    */
   if (packed_var) {
      /* Count array dimensions */
      const glsl_type *type = packed_var->type;
      unsigned dimensions = 0;
      while (type->is_array()) {
         type = type->fields.array;
         dimensions++;
      }

      /* Create a type list in reverse order (inner -> outer arrays) as this
       * is the order the IR works in.
       */
      const glsl_type **types_list = (const glsl_type **)
         rzalloc_array_size(mem_ctx, sizeof(glsl_type *), dimensions);
      unsigned order = dimensions;
      type = unpacked_var->type;
      while (type->is_array()) {
         types_list[--order] = type;
         type = type->fields.array;
      }

      /* Create a derefence for the packed var and clone the unpacked deref */
      unsigned inner_dimension = dimensions;
      bool has_vec_subscript = false;
      ir_dereference *packed = create_dereference(deref, &inner_dimension,
                                                  &has_vec_subscript);

      /* If the innermost array dereference is used to access vec components
       * rather than an array element remove it. This means we will end up
       * writting to all components with a shader like:
       *
       *    layout(location = 0, component = 1) patch out vec3 color;
       *    ...
       *    color[gl_InvocationID] = gl_InvocationID;
       *
       * The spec seems to support this. From the ARB_tessellation_shader
       * spec:
       *
       *    "Tessellation control shaders will get undefined results if one
       *    invocation reads a per-vertex or per-patch attribute written by
       *    another invocation at any point during the same phase, or if two
       *    invocations attempt to write different values to the same
       *    per-patch output in a single phase."
       *
       * FIXME: The text is a little unclear about what "attempt to write
       * different values to the same per-patch output" actually means. A spec
       * bug has been reported update once bug is resolved.
       * https://www.khronos.org/bugzilla/show_bug.cgi?id=1472
       */
      ir_dereference *cloned_deref;
      if (has_vec_subscript)
         cloned_deref = deref->as_dereference_array()->array->
                           as_dereference()->clone(this->mem_ctx, NULL);
      else
         cloned_deref = deref->clone(this->mem_ctx, NULL);

      /* If needed create extra derefs so we can copy all inner array elements
       * of a multi-dimensional array.
       */
      unsigned instruction_count;
      ir_dereference **packed_deref;
      ir_dereference **unpacked_deref;
      if (inner_dimension != 0) {
         instruction_count =
            types_list[inner_dimension - 1]->arrays_of_arrays_size();

         /* Create new lists to store the new instructions in */
         packed_deref = (ir_dereference **)
            rzalloc_array_size(mem_ctx, sizeof(ir_dereference *),
                               instruction_count);
         unpacked_deref = (ir_dereference **)
            rzalloc_array_size(mem_ctx, sizeof(ir_dereference *),
                               instruction_count);

         /* Pass in the outer array derefs that already exist */
         packed_deref[0] = packed;
         unpacked_deref[0] = cloned_deref;

         instruction_count =
            create_extra_array_dereference(inner_dimension - 1, types_list,
                                           packed_deref, unpacked_deref);
      } else {
         instruction_count = 1;
         packed_deref = &packed;
         unpacked_deref = &cloned_deref;
      }

      /* Wrap packed derefs in a swizzle and the create assignment */
      unsigned swizzle_values[4] = { 0, 0, 0, 0 };
      unsigned components =
         unpacked_var->type->without_array()->vector_elements;
      for (unsigned i = 0; i < components; ++i) {
         swizzle_values[i] = i + unpacked_var->data.location_frac;
      }

      for (unsigned i = 0; i < instruction_count; i++) {
         ir_swizzle *swiz = new(this->mem_ctx) ir_swizzle(packed_deref[i], swizzle_values,
                                              components);
         ir_assignment *assign;
         if (mode == ir_var_shader_out) {
            assign = new (this->mem_ctx) ir_assignment(swiz, unpacked_deref[i]);
         } else {
            assign = new (this->mem_ctx) ir_assignment(unpacked_deref[i], swiz);
         }
         new_instructions.push_tail(assign);
      }
   }
}

ir_visitor_status
lower_packed_varyings_tess_visitor::visit_leave(ir_dereference_array *ir)
{
   /* The array index is not the target of the assignment, so clear the
    * 'in_assignee' flag.  Restore it after returning from the array index.
    */
   const bool was_in_assignee = this->in_assignee;
   this->in_assignee = false;
   handle_rvalue(&ir->array_index);
   this->in_assignee = was_in_assignee;

   ir_rvalue *rvalue = ir;
   handle_rvalue(&rvalue);

   return visit_continue;
}

ir_visitor_status
lower_packed_varyings_tess_visitor::visit_leave(ir_assignment *ir)
{
   handle_rvalue(&ir->rhs);
   ir->rhs->accept(this);

   /* The normal rvalue visitor skips the LHS of assignments, but we
    * need to process those just the same.
    */
   ir_rvalue *lhs = ir->lhs;
   handle_rvalue(&lhs);
   ir->lhs->accept(this);

   if (ir->condition) {
      handle_rvalue(&ir->condition);
      ir->condition->accept(this);
   }

   return visit_continue;
}


void
lower_packed_varyings(void *mem_ctx, struct gl_shader_program *prog,
                      unsigned locations_used, ir_variable_mode mode,
                      gl_shader *shader, unsigned base_location,
                      bool disable_varying_packing, bool xfb_enabled,
                      bool has_enhanced_layouts)
{
   ir_function *main_func = shader->symbols->get_function("main");
   exec_list void_parameters;
   ir_function_signature *main_func_sig
      = main_func->matching_signature(NULL, &void_parameters, false);

   if (!(shader->Stage == MESA_SHADER_TESS_CTRL ||
         (shader->Stage == MESA_SHADER_TESS_EVAL &&
          mode == ir_var_shader_in))) {
      exec_list *instructions = shader->ir;
      exec_list new_instructions, new_variables;

      bool is_outer_array_vert_idx = false;
      if (mode == ir_var_shader_in &&
          shader->Stage == MESA_SHADER_GEOMETRY) {
         is_outer_array_vert_idx = true;
      }

      lower_packed_varyings_visitor visitor(mem_ctx, locations_used, mode,
                                            is_outer_array_vert_idx,
                                            &new_instructions,
                                            &new_variables,
                                            base_location,
                                            disable_varying_packing,
                                            xfb_enabled,
                                            has_enhanced_layouts);
      visitor.run(shader);
      if (mode == ir_var_shader_out) {
         if (shader->Stage == MESA_SHADER_GEOMETRY) {
            /* For geometry shaders, outputs need to be lowered before each
             * call to EmitVertex()
             */
            lower_packed_varyings_gs_splicer splicer(mem_ctx,
                                                     &new_instructions);

            /* Add all the variables in first. */
            main_func_sig->body.head->insert_before(&new_variables);

            /* Now update all the EmitVertex instances */
            splicer.run(instructions);
         } else {
            /* For other shader types, outputs need to be lowered at the end
             * of main()
             */
            main_func_sig->body.append_list(&new_variables);
            main_func_sig->body.append_list(&new_instructions);
         }
      } else {
         /* Shader inputs need to be lowered at the beginning of main() */
         main_func_sig->body.head->insert_before(&new_instructions);
         main_func_sig->body.head->insert_before(&new_variables);
      }
   } else {
      /* Build a hash table with all the varyings we can pack. For the
       * tessellation stages we only pack varyings that have location
       * and component layout qualifiers as packing varying without these
       * makes things much more difficult.
       */
      hash_table *varyings = _mesa_hash_table_create(NULL, _mesa_hash_pointer,
                                                     _mesa_key_pointer_equal);

      ir_variable **packed_varyings = (ir_variable **)
          rzalloc_array_size(mem_ctx, sizeof(*packed_varyings),
                                        locations_used);

      foreach_in_list(ir_instruction, node, shader->ir) {
         ir_variable *var = node->as_variable();
         if (var == NULL)
            continue;

         if (var->data.mode != mode ||
             var->data.location < (int) base_location ||
             !needs_lowering(var, has_enhanced_layouts, true))
            continue;

         const glsl_type *t;
         int location = var->data.location - base_location;
         if (var->data.patch) {
            location = var->data.location - VARYING_SLOT_PATCH0;
            t = var->type;
         } else {
            t = var->type->fields.array;
         }

         /* Clone the variable for program resource list before
          * it gets modified and lost.
          */
         if (!shader->packed_varyings)
            shader->packed_varyings = new (shader) exec_list;

         shader->packed_varyings->push_tail(var->clone(shader, NULL));

         /* Get the packed varying for this location or create a new one. */
         ir_variable *packed_var;
         if (packed_varyings[location]) {
            packed_var = packed_varyings[location];

            /* FIXME: Its possible to pack two different sized arrays together
             * and also packed varyings are not required to start at the same
             * location. However this would be difficult to do with the
             * current method of packing.
             */
            if (packed_var->data.location != var->data.location ||
                !check_for_matching_arrays(packed_var, var)) {
               unsigned packed_location = var->data.patch ?
                  packed_var->data.location - VARYING_SLOT_PATCH0 :
                  packed_var->data.location - base_location;
               const char *varying_mode =
                  var->data.mode == ir_var_shader_out ? "outputs" : "inputs";

               linker_error(prog, "Although allowed by the GLSL spec packing "
                            "varyings with different array types or starting "
                            "at different locations is not currently "
                            "supported in Mesa drivers for %s shader %s "
                            "(%s@%d vs %s@%d)\n.",
                            _mesa_shader_stage_to_string(shader->Stage),
                            varying_mode, packed_var->type->name,
                            packed_location, var->type->name, location);

               _mesa_hash_table_destroy(varyings, NULL);
               return;
            }
         } else {
            /* Create the new packed varying */
            packed_var = create_tess_packed_var(mem_ctx, var);
            var->insert_before(packed_var);
            packed_varyings[location] = packed_var;

            /* Add the var to the lookup table at all the locations it
             * consumes.
             */
            unsigned num_locs = t->count_attribute_slots(false);
            for (unsigned i = 0; i < num_locs; i++) {
               packed_varyings[location + i] = packed_var;
            }
         }

         /* Add to varyings the hash table with the old varying as a key, and
          * the packed varying as the data. This will be used later in the
          * visitor to look-up variables that need to be replaced.
          */
         _mesa_hash_table_insert(varyings, var, packed_var);

         /* Change the old varying into an ordinary global, dead code
          * elimination will clean this up for us later on.
          */
         assert(var->data.mode != ir_var_temporary);
         var->data.mode = ir_var_auto;
      }

      /* Find varying dereferences */
      /* Create instructions that copy varyings to/from temporaries */
      lower_packed_varyings_tess_visitor visitor(mem_ctx, varyings, mode);
      visitor.run(shader->ir);

      /* Insert instructions that copy varyings to/from temporaries */
      if (mode == ir_var_shader_out) {
         main_func_sig->body.append_list(&visitor.new_instructions);
      } else {
         main_func_sig->body.head->insert_before(&visitor.new_instructions);
      }

      _mesa_hash_table_destroy(varyings, NULL);
   }
}
