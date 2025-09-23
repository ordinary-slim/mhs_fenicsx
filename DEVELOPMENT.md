# Development roadmap

_This document tracks planned improvements and missing features._

## High priority

- [ ] **Support Dirichlet conditions in substepper**

  Currently, Dirichlet conditions are not supported when using substepping.
  Neumann and Robin conditions work as expected.

## Medium priority

- [ ] **Cap time step of fast problem at first micro-iteration**

  The advected subdomain requires a back length at least as large as the
  distance it will travel in a single time step. This is not guaranteed
  if the back length of the fast problem is too small, and no assertions
  guard against that. Currently, the user must be aware of this.

## Low Priority

- [ ] **Reset advected subdomain position in `pre_loop` of substepper**

  Without this step, advected subdomain methods cannot perform more than
  one substepping iteration. This is acceptable in most cases but should
  be corrected for completeness.

- [ ] **Unify `extract_cell_geo` implementations**

  The function is duplicated in `get_active_dofs_external` and
  `geometry_utils`. These should be unified to reduce code redundancy.

- [ ] **Mesh-independent forms in monolithic Robin driver**

  The current workflow compiles forms for a specific mesh and instantiates
  them for a given subdomain. A possible improvement would be compiling
  the forms abstracted from the mesh and "propagating" them (e.g. from
  slow to fast problem).

