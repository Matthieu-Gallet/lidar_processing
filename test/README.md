# Test Suite for LIDAR Processing Pipeline

This directory contains comprehensive tests for the LIDAR processing pipeline, including individual step tests and integration tests.

## Directory Structure

```
test/
├── run_tests.py                    # Quick test runner script
├── step1/
│   ├── test_step1_process_lidar.py # Test for LIDAR processing step
│   └── generate_test_lidar_data.py # Utility to generate synthetic LIDAR data
├── step2/
│   ├── test_step2_estimate_chm.py  # Test for CHM/FHD estimation step
│   └── generate_test_rasters.py    # Utility to generate synthetic raster data
├── step3/
│   ├── test_step3_chm_clustering.py # Test for clustering/classification step
│   └── generate_test_masks.py      # Utility to generate synthetic mask data
├── integration/
│   └── test_full_pipeline.py       # Full pipeline integration test
├── config/
│   ├── step1_test_config.json      # Test config for step 1
│   ├── step2_test_config.json      # Test config for step 2
│   ├── step3_test_config.json      # Test config for step 3
│   └── meta_test_config.json       # Meta config for integration tests
├── fixtures/
│   └── test_survey_data.csv        # Synthetic survey data
└── outputs/                        # Test output directories (created during tests)
    ├── step1/
    ├── step2/
    ├── step3/
    ├── integration/
    └── logs/
```

## Quick Start

### Run All Tests
```bash
# From the project root directory
python test/run_tests.py --all
```

### Run Individual Tests
```bash
# Test a specific step
python test/run_tests.py --step 1
python test/run_tests.py --step 2
python test/run_tests.py --step 3

# Test integration only
python test/run_tests.py --integration
```

### Test Management
```bash
# Show test information
python test/run_tests.py --info

# Setup test environment
python test/run_tests.py --setup

# Clean test outputs
python test/run_tests.py --clean

# Verbose output
python test/run_tests.py --all --verbose
```

## Individual Test Scripts

### Step 1: LIDAR Processing Test
- **Script**: `test/step1/test_step1_process_lidar.py`
- **Purpose**: Tests LIDAR point cloud processing, denoising, and classification
- **Generated Data**: Synthetic LIDAR point clouds (LAS/LAZ files)
- **Outputs**: Processed LIDAR files, classification results

### Step 2: CHM/FHD Estimation Test
- **Script**: `test/step2/test_step2_estimate_chm.py`
- **Purpose**: Tests CHM and FHD estimation from processed LIDAR data
- **Generated Data**: Synthetic raster data (GeoTIFF files)
- **Outputs**: CHM, FHD, DEM raster files

### Step 3: Clustering/Classification Test
- **Script**: `test/step3/test_step3_chm_clustering.py`
- **Purpose**: Tests vegetation clustering and classification
- **Generated Data**: Synthetic multichannel rasters and optional masks
- **Outputs**: Classification results, cluster analysis

### Integration Test
- **Script**: `test/integration/test_full_pipeline.py`
- **Purpose**: Tests the complete pipeline end-to-end
- **Features**: 
  - Runs individual tests first
  - Tests main pipeline executor
  - Analyzes all outputs
  - Provides comprehensive summary

## Test Data

### Synthetic Data Generation
Each test generates its own synthetic data to ensure consistency:

- **LIDAR Data**: Simulated point clouds with realistic height distributions
- **Raster Data**: Generated CHM, FHD, and DEM files with spatial patterns
- **Survey Data**: CSV with vegetation survey points and measurements
- **Mask Data**: Optional classification masks for testing robustness

### Test Configurations
All tests use dedicated test configurations in `test/config/`:
- Smaller data sizes for faster execution
- Simplified parameters for testing
- Output paths directed to test directories

## Expected Test Outcomes

### Successful Test Run
```
🧪 Running all LIDAR processing tests...
==================================================
✅ Step 1 test passed (45.2s)
✅ Step 2 test passed (32.1s) 
✅ Step 3 test passed (28.7s)
✅ Integration test passed (156.3s)

📊 Overall: 4/4 tests passed
🎉 All tests passed! Pipeline is ready.
```

### Test Outputs
After successful tests, you'll find:
- **test/outputs/step1/**: Processed LIDAR files
- **test/outputs/step2/**: CHM, FHD, DEM rasters
- **test/outputs/step3/**: Classification results
- **test/outputs/integration/**: Complete pipeline outputs
- **test/outputs/logs/**: Detailed log files

## Troubleshooting

### Common Issues

**ImportError: Module not found**
```bash
# Ensure you're in the project root directory
cd /path/to/lidar_processing
python test/run_tests.py --all
```

**Permission denied**
```bash
# Make scripts executable
chmod +x test/run_tests.py test/step*/test_*.py test/integration/test_*.py
```

**Test timeout**
- Individual step tests timeout after 5 minutes
- Integration test times out after 15 minutes
- Use `--verbose` flag to see real-time progress

### Memory Issues
If tests fail due to memory:
1. Reduce synthetic data sizes in test configs
2. Run individual tests instead of all at once
3. Monitor system resources during test execution

### Failed Tests
1. Check log files in `test/outputs/logs/`
2. Run individual tests with `--verbose` flag
3. Verify test data generation succeeded
4. Check configuration files for correct paths

## Test Maintenance

### Adding New Tests
1. Create test script in appropriate step directory
2. Follow naming convention: `test_stepX_*.py`
3. Include synthetic data generation
4. Update configurations if needed
5. Add to `run_tests.py` if needed

### Updating Test Data
1. Modify generation functions in individual test scripts
2. Update configurations in `test/config/`
3. Clean old outputs: `python test/run_tests.py --clean`
4. Re-run tests to verify changes

## Performance Benchmarks

Typical execution times on a standard development machine:
- **Step 1 Test**: 30-60 seconds
- **Step 2 Test**: 20-40 seconds  
- **Step 3 Test**: 20-40 seconds
- **Integration Test**: 2-5 minutes total

Times may vary based on:
- System specifications
- Data sizes in test configs
- I/O performance
- Available memory

## Integration with CI/CD

The test suite is designed for integration with continuous integration:

```bash
# CI script example
python test/run_tests.py --setup
python test/run_tests.py --all
exit_code=$?
python test/run_tests.py --clean
exit $exit_code
```

All scripts return appropriate exit codes:
- `0`: All tests passed
- `1`: One or more tests failed

---

For questions or issues with the test suite, check the main project documentation or examine individual test scripts for detailed implementation.
