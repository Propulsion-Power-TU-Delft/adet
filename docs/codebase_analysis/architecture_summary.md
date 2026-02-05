# ADeT Architecture - Quick Reference

## Core Pattern
**Equation-Oriented (EO) Modeling** with multiple solver backends

## The 4 Pillars

| Pillar | File | Size | Purpose |
|--------|------|------|---------|
| **FlowNode** | `node.py` | 235 | Encapsulates complete thermo-kinematic state (6 containers) |
| **EquationBase** | `equations/base_equation.py` | 233 | Abstract residual equations with symbolic support |
| **SystemAssembler** | `assembly.py` | 1,153 | Orchestrates equation compilation + multi-backend support |
| **ComponentNetwork** | `components/network.py` | 128 | Composes multi-stage turbomachinery systems |

## Architecture Highlights

```
Physics (Equations)
    ↓ (symbolic)
EquationBase (validation, argument parsing)
    ↓ (registration)
SystemAssembler (system assembly, scaling, unit checking)
    ↓ (compilation choice)
    ├─→ CasadiSystem (Newton-Raphson, code generation)
    └─→ JaxSystem (autodiff, functional)
    ↓ (execution)
ComponentNetwork (multi-stage composition)
    ↓ (evaluation)
FlowNode (stores solution across stages)
```

## Key Files by Importance

### Critical (Must Understand)
1. `node.py` - Data model for flow state
2. `assembly.py` - System compilation engine
3. `equations/base_equation.py` - Equation abstraction
4. `components/network.py` - Component orchestration

### Important (Extend for Research)
5. `losses/profile.py` - Loss correlations (706 LOC!)
6. `fluid/casadi_eos.py` - Thermodynamic integration via CasADi callbacks
7. `equations/fundamental.py` - Turbomachinery equations
8. `components/blade_row.py` - Stage representation

### Supporting
9. `registries.py` - Singleton pattern for defaults
10. `variables.py` - Variable container system
11. `tools/` - Utilities (CoolProp, strings, plotting)

## Code Size Distribution

```
assembly.py                 1,153 LOC  ← MONOLITHIC (refactor candidate)
losses/profile.py             706 LOC  ← COMPLEX (CasADi-only loss)
diagnostics.py                556 LOC
geometry.py                   438 LOC
variables.py                  402 LOC
components/blade_row.py       328 LOC
tools/coolprop_utils.py       302 LOC
fluid/casadi_eos.py           287 LOC
registries.py                 286 LOC
equations/fundamental.py      255 LOC
node.py                       235 LOC
equations/base_equation.py    233 LOC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~8,500 LOC
```

## Architectural Strengths ✅

1. **Separation of Concerns** - Physics, solving, thermodynamics separate
2. **Backend Flexibility** - CasADi + JAX support with conversion
3. **Automatic Differentiation** - Jacobians computed automatically
4. **Unit Safety** - Pint integration with unit checking
5. **Extensibility** - Plugin points for equations, components, losses

## Architectural Weaknesses 🔴

1. **SystemAssembler Monolith** - 1,153 LOC violates SRP
   - Should split into: ArgumentMapper, ConstraintManager, ScalingManager, UnitValidator
   
2. **No Test Suite** - pytest exists but no tests visible
   - Need minimum 60% coverage
   
3. **FlowNode ID Pattern** - Uses class-level counter (thread-unsafe)
   - Should accept node IDs from SystemAssembler
   
4. **Loss Model Integration** - Incomplete (`_add_loss_parameters` not implemented)
   - Loss models as equations ≠ loss model objects
   
5. **CasADi-Only Features** - DentonProfileLoss incompatible with JAX
   - Limits backend flexibility
   
6. **Equation Counting** - Fragile heuristics (AST + argument injection)
   - Silent fallbacks hide errors
   
7. **Suspended Validations** - Well-posedness check disabled
   - Harder to debug misconfigured systems
   
8. **Deprecated Code** - 100KB of unused components
   - Maintenance burden, confuses users

## Key TODOs in Codebase

```python
# assembly.py:65
# TODO: This class is quite heavy, it is probably a good idea
# to break it down into more manageable components

# assembly.py:305
# TODO: Well posedness check suspended for now
# Reasons: Dynamic argument choice difficult, solver throws error anyway

# components/blade_row.py:54
# TODO: Fix this for multiple formulations interacting

# equations/fundamental.py:129
# TODO: Add differential equation for streamline curvature

# base_equation.py:34
# TODO: Move scaling factor
```

## Recent Refactoring History

From git log:
- **34543f5**: Refactor Denton profile loss model and clean up main solver
- **b8b69db**: Update blade row equations and profile loss models
- **6221b13**: Add deepcopy method for eos callback, debug profile loss

→ Profile loss model is actively being refined

## Dependencies

### Heavy (Complex Integration)
- **CasADi** 3.7.2+ (symbolic, solver)
- **CoolProp** 7.1.0+ (thermodynamics)
- **JAX** (autodiff)

### Standard
- Pint, SciPy, SymPy, Matplotlib, Bezier, Optimistix

## Design Patterns Used

| Pattern | Where | Notes |
|---------|-------|-------|
| **Abstract Factory** | BaseComponent | Creates equation instances |
| **Singleton** | Registries | Global defaults with overrides |
| **Strategy** | CasadiSystem/JaxSystem | Pluggable backends |
| **Visitor** | override_operators | Traverse expressions |
| **Decorator** | @thermo_property | Property decorators |
| **Mixin** | GasPropertiesMixin | Add derived properties |
| **Generic** | ComponentNetwork[T] | Type-parameterized |
| **Template Method** | BaseComponent | Subclass defines base_equations |

## Performance Characteristics

- **Equation Setup**: O(n_eqs × n_nodes) for registration
- **Solving**: Depends on backend (Newton-Raphson for CasADi)
- **CoolProp**: Can be bottleneck for complex gases
- **Spanwise**: Vectorized across stations (num_span dimension)

## Testing Gaps

No visible test suite despite:
- pytest in dependencies
- ~8,500 LOC of code
- Complex interactions (equations → assembly → solving)

**Needed**:
- Unit tests for equation registration
- Integration tests for system building
- Solver convergence tests
- Loss model validation tests

## Suggested Next Steps for Research Use

1. **Understand EO Pattern** - Read equations/fundamental.py
2. **Try Example** - Run main.py or examples/
3. **Add Custom Equation** - Subclass EquationBase
4. **Profile Your System** - Use diagnostics.py
5. **Optimize Scaling** - Adjust registries.py defaults

## To Extend ADeT

| Goal | How |
|------|-----|
| Add equation | Subclass `EquationBase`, define `residual()` |
| Add component | Subclass `BaseComponent`, define `base_equations` |
| Add loss model | Subclass `LossModel`, integrate into equations |
| Add fluid model | Subclass `FluidModel`, integrate into FluidSettings |
| Switch solver | Use `CasadiSystem` or `JaxSystem` |
| Add custom variable | Extend `VariableContainer` |

## Code Quality Score: 6.2/10

| Aspect | Score | Status |
|--------|-------|--------|
| Modularity | 7/10 | Good, but SystemAssembler too large |
| Testability | 3/10 | 🔴 No test suite |
| Documentation | 5/10 | Moderate docstrings, missing architecture guide |
| Error Handling | 6/10 | Good explicit errors, some silent fallbacks |
| Type Safety | 6/10 | Basic hints, intentional gaps |
| Code Clarity | 7/10 | Generally readable |
| Extensibility | 8/10 | Many plugin points |
| Performance | 7/10 | Good compilation, CoolProp can bottleneck |
| Research Fit | 6/10 | Excellent for exploration, needs hardening |

## Path to Production (3-4 Weeks)

1. **Week 1**: Add test suite + refactor SystemAssembler
2. **Week 2**: Architecture docs + API stabilization  
3. **Week 3**: Complete loss model integration + examples
4. **Week 4**: Review + hardening

---

**For Detailed Analysis**: See `ARCHITECTURE_ANALYSIS.md`
