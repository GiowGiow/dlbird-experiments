# SpecKit Specifications Directory

This directory contains all feature specifications, implementation plans, and experimental documentation following the SpecKit methodology.

## Directory Structure

```
.specify/specs/
├── README.md                       # This file - navigation guide
├── REORGANIZATION_SUMMARY.md       # Why and how repo was reorganized
├── 001-validation-phase/           # Initial validation and root cause analysis
│   ├── spec.md                     # Validation requirements and user stories
│   ├── plan.md                     # Validation execution plan
│   ├── quickstart.md               # Quick start guide for validation
│   └── artifacts/                  # Validation outputs (moved from root)
│       ├── recommended_class_weights.json
│       ├── class_distribution_*.png
│       └── feature_*.png
├── 002-phase1-critical-fixes/      # Class weights + normalization
│   ├── spec.md                     # Phase 1 functional specification
│   ├── plan.md                     # Implementation plan with code examples
│   ├── tasks.md                    # Ordered task breakdown
│   ├── quickstart.md               # Immediate action steps
│   └── artifacts/                  # Phase 1 experiment results
│       └── (results will be saved here)
├── 003-phase2-feature-engineering/ # Mel-spectrograms + augmentation (future)
├── 004-phase3-architecture/        # Pretrained models (future)
├── 005-phase4-advanced/            # Multi-modal fusion (future)
└── archive/                        # Old CAPS-LOCK named docs (reference only)
    ├── README.md                   # Guide to archived files
    ├── old-validation-docs/        # Pre-SpecKit validation docs
    ├── old-planning-docs/          # Pre-SpecKit planning docs
    └── old-implementation-docs/    # Various summaries and results
```

## Specification Workflow

Each feature/phase follows this workflow:

1. **Specification** (`spec.md`)
   - Define WHAT to build (tech-stack agnostic)
   - User stories and functional requirements
   - Success criteria and acceptance criteria
   - Created BEFORE technical planning

2. **Clarification** (optional)
   - Use `/speckit.clarify` to refine requirements
   - Document questions and answers in spec
   - Ensure completeness before planning

3. **Planning** (`plan.md`)
   - Define HOW to build (technical details)
   - Architecture decisions and tech stack
   - File modifications and implementation steps
   - Research findings and library versions

4. **Task Breakdown** (`tasks.md`)
   - Generate from plan using `/speckit.tasks`
   - Ordered tasks with dependencies
   - Parallel execution markers [P]
   - Checkpoint validations

5. **Implementation**
   - Follow tasks.md systematically
   - Document experiments in artifacts/
   - Track progress against success criteria

6. **Validation & Completion**
   - Verify all acceptance criteria met
   - Document results and learnings
   - Update status in spec.md header

## Document Templates

Templates are available in `.specify/templates/`:
- `spec-template.md` - For creating new specifications
- `plan-template.md` - For implementation plans
- `tasks-template.md` - For task breakdowns
- `checklist-template.md` - For validation checklists

## Current Status

### ✅ Completed
- **001-validation-phase**: Root cause analysis complete
  - Severe class imbalance identified (1216:1)
  - Missing feature normalization detected
  - Validation artifacts generated

### 📋 Ready for Implementation
- **002-phase1-critical-fixes**: Spec and plan complete
  - Class-weighted loss implementation
  - Feature normalization
  - Expected F1-macro: 0.109 → 0.25-0.35

### 🔮 Future
- **003-phase2-feature-engineering**: Mel-spectrograms + augmentation
- **004-phase3-architecture**: Pretrained models
- **005-phase4-advanced**: Multi-modal fusion

## Naming Convention

Specs are numbered sequentially with descriptive names:
- Format: `NNN-descriptive-name/`
- N = 3-digit number (001, 002, etc.)
- descriptive-name = kebab-case feature description
- Keep names concise but clear

## Migration Notes

The following documents have been organized into specs:

**From root to 001-validation-phase/**:
- VALIDATION_SUMMARY_EXECUTIVE.md → spec.md (integrated)
- VALIDATION_RESULTS.md → plan.md (integrated)
- INDEX_VALIDATION.md → quickstart.md
- artifacts/validation/* → artifacts/

**From root to 002-phase1-critical-fixes/**:
- IMPLEMENTATION_PLAN.md (Phase 1) → plan.md
- QUICK_START_IMPLEMENTATION.md → quickstart.md

**Remaining in root** (project-level):
- README.md - Project overview
- QUICKSTART.md - General quick start
- STATUS.md - Current project status
- INDEX_PLANNING.md - Master navigation (updated to reference specs/)

## References

- SpecKit Documentation: https://specifyapp.github.io/
- Constitution: `.specify/memory/constitution.md`
- Templates: `.specify/templates/`
