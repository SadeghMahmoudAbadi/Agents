A. Per-file Analysis

**1. EPA_Project/EPA_program.py**
1. Filename & path: EPA_Project/EPA_program.py
2. File Summary: Serves as the main CLI entry point for the EPA transformer design application, guiding users through interactive input collection for various transformer types and instantiating appropriate classes to compute and output optimal designs.
3. Key Implementation Notes:
   - Implements stateful flag-based input workflow for transformer selection (CT, CTR, CTW, LP, CL.X, Tap, Multi Core, Design) with validation and terminal clearing.
   - Collects parameters like currents, power, error class, dimensions (OD/ID/H), insulation values, and instantiates specialized Transformer subclasses based on type/error class.
   - Handles special cases like LP class input, CL.X Vk/Rct or coefficient-based variants, tap changer primary lists, and design mode with direct result computation.
   - Integrates multi-core via external main() call and tap processing via tap_func(); outputs design lists or single results.
4. Skills Demonstrated:
   - Interactive CLI development with input validation and error handling
   - Modular OOP integration across imported classes
   - State machine design for multi-step user workflows
   - Cross-platform terminal utilities (os.system for clear/cls)

**2. EPA_Project/class_fs.py**
1. Filename & path: EPA_Project/class_fs.py
2. File Summary: Defines FS-class error handling subclasses (CTfs, CTRfs, CTWfs, LPfs) inheriting from Transformer, customizing net diameter selection, dimension checks, and error evaluation for phase saturation and correction logic.
3. Key Implementation Notes:
   - CTfs overrides set_net_list for secondary current-based diameters and evaluate_error with class-specific saturation checks/corrections (turns/net double correction).
   - CTRfs/CTWfs extend with mould fitting (CTR_mould_list/CTW_mould_list), flesh ratio validation, and Rct computation via design_interface.
   - LPfs generates routine designs from LP_core_list, checks against LP_mould_list, and evaluates without phase conditions for higher error classes.
   - All apply B-range filtering, price/validity checks, and append to possible/accepted_designs based on corrected current_120 thresholds.
4. Skills Demonstrated:
   - OOP inheritance and polymorphism for variant-specific logic
   - Precision engineering calculations (saturation, error correction, mould fitting)
   - Inventory-driven design generation and filtering
   - Domain-specific validation (flesh ratios, Rct resistance)

**3. EPA_Project/class_p.py**
1. Filename & path: EPA_Project/class_p.py
2. File Summary: Provides nPm-class error subclasses (CTnpm, CTRnpm, CTWnpm, LPnpm) extending Transformer with tailored net lists, error evaluation including turns/double corrections, and mould/dimension checks.
3. Key Implementation Notes:
   - CTnpm customizes set_net_list for high primary currents and evaluate_error with saturation/B-range corrections for 10P3 or standard nPm classes.
   - CTRnpm/CTWnpm add mould fitting from CTR/CTW lists, Rct checks, and flesh validation.
   - LPnpm generates routine designs via LP_core_list/H/net loops, validates against LP_mould_list.
   - All filter designs by B-range, price >0, dimensions, and accept based on corrected current_120 > -error_class.
4. Skills Demonstrated:
   - Advanced OOP subclassing for error-class variants
   - Numerical optimization (nearest net diameters via NumPy)
   - Mechanical fit validation (moulds, flesh proportions)
   - Error correction algorithms (turns adjustment, net dual-diameter)

**4. EPA_Project/class_x.py**
1. Filename & path: EPA_Project/class_x.py
2. File Summary: Implements CL.X transformer classes (ClassX, SpecialX) with net list setup, B computation variants, dimension/flesh checks, and basic evaluation/output for simulation results.
3. Key Implementation Notes:
   - ClassX sets broad net lists by secondary current, uses compute_B_X, checks dimensions/Rct/flesh (3x ratio), evaluates designs simply.
   - SpecialX overrides B via compute_B_X_2 and shares checks; supports interactive sorting/output of accepted designs by price/B/error.
   - Includes fit validation without moulds, phase-independent logic.
4. Skills Demonstrated:
   - Specialized subclassing for voltage/resistance constraints
   - Interactive result presentation (sorting, pagination)
   - Geometric validation (final dimensions, flesh ratios)
   - NumPy-based nearest-neighbor selection

**5. EPA_Project/clean_data.py**
1. Filename & path: EPA_Project/clean_data.py
2. File Summary: Preprocesses transformer simulation data from CSV, standardizing net diameters, adding gross/resistivity/power factor features, and deduplicating for ML-ready dataset.
3. Key Implementation Notes:
   - Defines net_diameter_list and net_list dict with gross/resistivity values; nearest-net via NumPy argmin.
   - Loads data.csv, filters features, applies transformations (nearest net, gross/resistivity, power_factor >=2.5?0.8:1), moves current_120 last, drops duplicates, saves dataset.csv.
4. Skills Demonstrated:
   - ETL pipelines with Pandas/NumPy
   - Domain data cleaning (net standardization, feature engineering)
   - CSV I/O and deduplication
   - Lookup table integration for material properties

**6. EPA_Project/ct_interface.py**
1. Filename & path: EPA_Project/ct_interface.py
2. File Summary: Automates legacy DOSBox-based CT simulator via file I/O, pyautogui keystrokes, and win32gui for input generation, execution, and result parsing.
3. Key Implementation Notes:
   - Defines constants (wire diameters, BH curve, material props); writes CT.DAT inputs (currents/power/turns/core/wire/BH).
   - Manages DOSBox (open/activate via pygetwindow, cd/ct/alt-eval/quit); parses CT000002.OUT for current/phase errors at 20/100/120%.
   - Handles language switch, retries on failure, nearest wire index.
4. Skills Demonstrated:
   - Legacy software automation (pyautogui, win32api, subprocess)
   - File-based simulation interfacing (input/output parsing)
   - Cross-platform window/process management
   - Engineering constant management (BH curves, material densities)

**7. EPA_Project/desktop_gui_app.py**
1. Filename & path: EPA_Project/desktop_gui_app.py
2. File Summary: Builds a comprehensive PyQt6 GUI for EPA design with dynamic forms per transformer type, multi-core config, settings/inventory management, threaded backend processing, and results display.
3. Key Implementation Notes:
   - DynamicFormWidget generates fields (spinboxes/combos/checks) by type (CT/LP/CL.X/TAP/Design); BackendWorker threads model instantiation/train/tap_func/price_analysis.
   - SettingsWindow loads/saves JSON inventory/settings with checkboxes/inputs for sections/elements; MultiCoreWindow configs per-core forms.
   - OutputWindow scrolls formatted results; DPI/high-DPI handling via env vars/ctypes.
   - Integrates all classes, sorts results by price, Excel price via xlwings indirectly.
4. Skills Demonstrated:
   - Modern GUI development (PyQt6, dynamic forms, threading, QThread signals)
   - JSON config/inventory management
   - Multi-window dialogs (settings, multi-core, results)
   - Cross-platform DPI/window awareness

**8. EPA_Project/inventory_tools.py**
1. Filename & path: EPA_Project/inventory_tools.py
2. File Summary: Simple JSON loader/saver for inventory/config data, with a dummy custom() call for testing.
3. Key Implementation Notes:
   - Loads/saves config.json; custom() reloads/saves unchanged inventory.
4. Skills Demonstrated:
   - JSON persistence for app data
   - Modular data access utilities

**9. EPA_Project/multi_core.py**
1. Filename & path: EPA_Project/multi_core.py
2. File Summary: Handles multi-core transformer designs by generating per-core accepted designs, recursively finding best combinations within tolerance/H constraints, and computing total price.
3. Key Implementation Notes:
   - run_multi trains each transformer, uses find_best_combination (backtracking) to select combos where ID/OD max-min < tolerance and sum(H) <= order['H'].
   - calculate_price sums design prices; CLI main() for testing.
4. Skills Demonstrated:
   - Recursive combinatorial optimization
   - Tolerance-constrained design selection
   - Multi-component system integration

**10. EPA_Project/price_analysis.py**
1. Filename & path: EPA_Project/price_analysis.py
2. File Summary: Automates Excel-based price/material analysis via xlwings, populating templates with design/order data, barcodes, weights, ITZ adjustments, and extracting final costs.
3. Key Implementation Notes:
   - Copies templates (Temp LV*.xlsx), computes values (dimensions, lugs, nets, H_count via DP min_H_sum, order strings), single-session writes/reads.
   - Handles modes (routine/mandrel/custom), ITZ/shellac, exceptions (large ID/weight); Persian date via jdatetime.
   - analyse_price returns (price, material); main() tests.
4. Skills Demonstrated:
   - Excel automation (xlwings batch ops)
   - Dynamic programming for bin packing (H_count)
   - BOM/price calculation (barcodes, weights, adjustments)
   - Template-driven reporting

**11. EPA_Project/tap_trans.py**
1. Filename & path: EPA_Project/tap_trans.py
2. File Summary: Processes tap changer designs by simulating errors across primary current list via ct_interface, storing per-primary current_120 in designs.
3. Key Implementation Notes:
   - tap_func opens DOSBox, scales power per primary, simulates via load_design_error/simulate_design_error, deletes original current_120.
4. Skills Demonstrated:
   - Multi-variant simulation (tap primaries)
   - Legacy tool batch automation
   - Data augmentation for variable loads

**12. EPA_Project/transformer.py** (Note: Input lists as 11 but content provided; treated as 12th per sequence)
1. Filename & path: EPA_Project/transformer.py
2. File Summary: Base Transformer class orchestrates design generation (routine/mandrel/custom), simulation via ct_interface, error evaluation/correction, price computation, and training pipeline.
3. Key Implementation Notes:
   - Loads inventory, sets net lists, generates designs (routine_core_list/H/net loops or fixed-ID OD/H), checks dimensions/B/Rct/flesh/price.
   - Simulates via DOSBox (load/save data.csv), corrects turns/net dual-diameter/phase, trains hierarchically (routine->mandrel->custom).
   - get_result for design mode; integrates price_analysis.
4. Skills Demonstrated:
   - Full design-to-evaluation pipeline
   - Hierarchical search (routine fallback to custom)
   - Simulation caching (CSV lookup)
   - Correction heuristics (turns, dual-net, phase)

B. Project-level Summary
1. Elevator Pitch: Developed EPA, a comprehensive Python application for automated current transformer design, simulation, optimization, and pricing across multiple types including multi-core and tap variants.
2. Project Overview: Core Transformer base class and subclasses (nPm/FS/CL.X) generate/evaluate designs from inventory-constrained cores/nets/H, simulate performance via DOSBox-automated legacy CT tool (caching results in CSV), apply corrections for errors/phase/saturation, and filter by B-range/dimensions/Rct/moulds. CLI (EPA_program.py) and PyQt6 GUI (desktop_gui_app.py) handle inputs; multi_core.py optimizes core combinations under tolerances; price_analysis.py uses xlwings for Excel BOM/pricing. Data flows from user specs -> design enum -> simulation -> correction -> acceptance -> sorted output/Excel report.
3. Impact & Outcomes:
   - Automates complex transformer engineering from specs to manufacturable designs, reducing manual iteration.
   - Enables rapid prototyping of variants (tap/multi-core) with precise error compliance and cost optimization.
   - Integrates legacy simulation with modern GUI/Excel for production-ready workflows.
   - Generates datasets for potential ML enhancement via clean_data.py.
4. Tech Stack Snapshot: Python, NumPy, Pandas, PyQt6, xlwings, pyautogui/win32gui/pygetwindow, JSON, DOSBox automation, Excel templates

C. Developer Skill Summary (Resume-ready)
1. Top Skills:
   - Transformer design automation (OOP subclasses for CT/CTR/CTW/LP/CL.X/nPm/FS)
   - Legacy simulation integration (DOSBox/pyautogui/ct_interface)
   - GUI development (PyQt6 dynamic forms, threading, multi-dialogs)
   - Excel automation (xlwings batch pricing/BOM)
   - Combinatorial optimization (backtracking multi-core, DP H-packing)
   - Data pipelines (Pandas ETL, simulation caching)
   - Engineering calculations (B-field, error correction, dimensions/Rct)
   - Inventory/config management (JSON, dynamic settings)
2. Resume Bullets:
   - Engineered full-stack transformer design system integrating CLI/GUI inputs, inventory-driven generation, DOSBox simulation, and Excel pricing for optimal compliant designs.
   - Designed hierarchical OOP architecture with 10+ subclasses handling error classes, tap/multi-core variants, and corrections yielding production-ready outputs.
   - Developed PyQt6 desktop app with dynamic forms, threaded processing, and inventory settings enabling user-friendly multi-core/tap workflows.
   - Optimized price analysis via single-session xlwings automation and DP algorithms, computing BOMs/weights/barcodes with ITZ adjustments.
   - Implemented simulation caching and error correction (turns/dual-net/phase) reducing computation time while ensuring accuracy.
   - Built combinatorial solver for multi-core designs under tolerance/H constraints, selecting minimal-price combinations.
3. One-line LinkedIn Headline suggestion: Python Engineer | Automated Transformer Design GUI | OOP Simulation & Pricing Optimization Expert