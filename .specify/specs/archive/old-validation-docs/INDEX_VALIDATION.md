# Validation Framework Index

**Created**: December 4, 2025  
**Purpose**: Guide to all validation documentation and scripts  
**Status**: ✅ Complete

---

## 📚 Documentation Overview

This validation framework consists of multiple documents serving different purposes:

| Document | Purpose | When to Use | Size |
|----------|---------|-------------|------|
| **`QUICK_START_VALIDATION.md`** | Quick orientation | First read, getting started | 3 KB |
| **`README_VALIDATION.md`** | Comprehensive guide | Detailed instructions, troubleshooting | 10 KB |
| **`VALIDATION_CHECKLIST.md`** | Detailed checklist | Systematic validation, sign-off | 26 KB |
| **`QA_SUMMARY.md`** | Executive summary | High-level overview, planning | 12 KB |
| **`INDEX_VALIDATION.md`** | This file | Navigation and reference | 2 KB |

---

## 🎯 Choose Your Path

### Path 1: I Want to Start Immediately
→ Read: **`QUICK_START_VALIDATION.md`**  
→ Run: `python scripts/run_all_validations.py`  
→ Time: 5-10 minutes

### Path 2: I Want Full Understanding  
→ Read: **`README_VALIDATION.md`** (comprehensive guide)  
→ Read: **`VALIDATION_CHECKLIST.md`** (detailed checklist)  
→ Run: Individual validation scripts  
→ Time: 1-2 hours

### Path 3: I Need Executive Overview
→ Read: **`QA_SUMMARY.md`** (high-level summary)  
→ Review: Key findings and recommendations  
→ Time: 15 minutes

---

## 🔧 Scripts Reference

| Script | Purpose | Runtime | Outputs |
|--------|---------|---------|---------|
| `validate_data.py` | Data integrity | 1-2 min | Terminal report |
| `validate_features.py` | Feature quality | 2-3 min | Terminal + plots |
| `validate_class_balance.py` | Class distribution | 2-3 min | Terminal + plots + JSON |
| `run_all_validations.py` | Run all | 5-10 min | Comprehensive report |

---

## 📊 Outputs Location

All validation outputs are saved to:
```
artifacts/validation/
├── feature_distributions.png
├── sample_features_0.png
├── sample_features_1.png
├── sample_features_2.png
├── class_distribution_audio_dataset.png
├── class_distribution_image_dataset.png
└── recommended_class_weights.json
```

---

## 🚦 Decision Flow

```
Start Here
    ↓
Read QUICK_START_VALIDATION.md (5 min)
    ↓
Run: python scripts/run_all_validations.py (10 min)
    ↓
Review outputs in artifacts/validation/
    ↓
    ├─ All Pass? → Review QA_SUMMARY.md → Proceed
    ├─ Warnings? → Read README_VALIDATION.md → Investigate
    └─ Failures? → Use VALIDATION_CHECKLIST.md → Fix issues
```

---

## 📖 Document Descriptions

### QUICK_START_VALIDATION.md
- **What**: Minimal guide to get started
- **Contains**: Commands, file list, expected outcomes
- **Best for**: First-time users, quick reference

### README_VALIDATION.md  
- **What**: Complete how-to guide
- **Contains**: Step-by-step instructions, troubleshooting, decision tree
- **Best for**: Detailed execution, problem-solving

### VALIDATION_CHECKLIST.md
- **What**: Systematic validation checklist
- **Contains**: 100+ checks organized by category, code examples
- **Best for**: Thorough validation, sign-off, documentation

### QA_SUMMARY.md
- **What**: High-level summary
- **Contains**: Critical issues, recommendations, experiment plan
- **Best for**: Planning, team communication, executive review

---

## 💡 Quick Commands

```bash
# Navigate to project
cd /home/giovanni/ufmg/speckitdlbird

# Run all validations (recommended)
python scripts/run_all_validations.py

# Run individual checks
python scripts/validate_data.py
python scripts/validate_features.py
python scripts/validate_class_balance.py

# View results
ls -lh artifacts/validation/
cat artifacts/validation/recommended_class_weights.json
```

---

## ✅ Checklist for First-Time Use

- [ ] Read `QUICK_START_VALIDATION.md` (5 min)
- [ ] Run `python scripts/run_all_validations.py` (10 min)
- [ ] Review terminal output for red flags
- [ ] Check `artifacts/validation/` for plots
- [ ] Review `VALIDATION_CHECKLIST.md` section by section
- [ ] Document findings and next steps
- [ ] Mark validation complete before experiments

---

## 🎓 Learning Path

**New User (Day 1)**:
1. QUICK_START_VALIDATION.md
2. Run validations
3. Review outputs

**Deep Dive (Day 2-3)**:
1. README_VALIDATION.md
2. VALIDATION_CHECKLIST.md
3. Individual validations

**Planning (Day 3-4)**:
1. QA_SUMMARY.md
2. Document findings
3. Plan experiments

---

## 📞 Support

**If validations fail**: Check `README_VALIDATION.md` troubleshooting section  
**If unclear**: Review `VALIDATION_CHECKLIST.md` for detailed explanations  
**For planning**: Use `QA_SUMMARY.md` recommendations  

---

**Start Here**: `QUICK_START_VALIDATION.md`  
**Questions**: `README_VALIDATION.md`  
**Details**: `VALIDATION_CHECKLIST.md`  
**Overview**: `QA_SUMMARY.md`
