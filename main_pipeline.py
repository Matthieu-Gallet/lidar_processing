#!/usr/bin/env python3
"""
LIDAR Processing Pipeline - Main Execution Script
==================================================

This script orchestrates the complete LIDAR data processing pipeline:
1. Step 1: Raw LIDAR data processing and initial CHM generation
2. Step 2: Multi-quantile CHM computation and FHD analysis
3. Step 3: Vegetation classification using clustering algorithms

Author: Herbiland Project
Date: 2025-01-31
"""

import os
import sys
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import argparse


class PipelineExecutor:
    """Main class for executing the LIDAR processing pipeline"""

    def __init__(self, meta_config_path="config/meta_config.json"):
        """Initialize the pipeline executor"""
        self.meta_config_path = meta_config_path
        self.meta_config = self.load_meta_config()
        self.setup_logging()
        self.start_time = None
        self.step_times = {}

    def load_meta_config(self):
        """Load the meta configuration file"""
        try:
            with open(self.meta_config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Meta configuration file not found: {self.meta_config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing meta configuration: {e}")
            sys.exit(1)

    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.meta_config.get("logging", {})
        log_dir = self.meta_config["paths"]["log_directory"]

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(log_dir, "pipeline_execution.log")
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                (
                    logging.StreamHandler()
                    if log_config.get("console_output", True)
                    else logging.NullHandler()
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def print_banner(self):
        """Print pipeline banner"""
        meta = self.meta_config["meta_config"]
        print("=" * 80)
        print(f"🚀 {meta['project_name']} v{meta['version']}")
        print(f"📝 {meta['description']}")
        print(f"👤 Author: {meta['author']}")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def print_step_info(self, step_key, step_config):
        """Print detailed information about a processing step"""
        print(f"\n📋 {step_config['description']}")
        print(f"⚙️  Script: {step_config['script_file']}")
        print(f"🔧 Config: {step_config['config_file']}")
        print(f"⏱️  Estimated time: {step_config['estimated_time_minutes']} minutes")
        print(f"💾 Memory requirement: {step_config['memory_requirement_gb']} GB")

        if step_config["depends_on"]:
            print(f"🔗 Dependencies: {', '.join(step_config['depends_on'])}")
        else:
            print("🔗 Dependencies: None")

    def check_dependencies(self, step_key, step_config):
        """Check if step dependencies are satisfied"""
        for dep in step_config["depends_on"]:
            if dep not in self.step_times:
                self.logger.error(f"Dependency {dep} not completed for {step_key}")
                return False
        return True

    def check_config_file(self, config_file):
        """Check if configuration file exists and is valid"""
        if not os.path.exists(config_file):
            self.logger.error(f"Configuration file not found: {config_file}")
            return False

        try:
            with open(config_file, "r") as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file {config_file}: {e}")
            return False

    def check_script_file(self, script_file):
        """Check if script file exists"""
        if not os.path.exists(script_file):
            self.logger.error(f"Script file not found: {script_file}")
            return False
        return True

    def execute_step(self, step_key, step_config):
        """Execute a single pipeline step"""
        print(f"\n🎯 Executing {step_key.upper()}")
        print("-" * 50)

        step_start_time = time.time()

        # Check prerequisites
        if not self.check_config_file(step_config["config_file"]):
            return False

        if not self.check_script_file(step_config["script_file"]):
            return False

        if not self.check_dependencies(step_key, step_config):
            return False

        try:
            # Execute the step script
            cmd = [sys.executable, step_config["script_file"]]
            self.logger.info(f"Executing command: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            step_duration = time.time() - step_start_time
            self.step_times[step_key] = step_duration

            print(
                f"✅ {step_key.upper()} completed successfully in {step_duration:.2f} seconds"
            )
            self.logger.info(f"{step_key} completed in {step_duration:.2f} seconds")

            if self.meta_config["pipeline_settings"]["verbose"]:
                print("📤 Script output:")
                print(result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ {step_key.upper()} failed with return code {e.returncode}")
            self.logger.error(f"{step_key} failed: {e}")

            if e.stderr:
                print(f"❌ Error output:\n{e.stderr}")
                self.logger.error(f"{step_key} stderr: {e.stderr}")

            if not self.meta_config["pipeline_settings"]["continue_on_error"]:
                return False

        except Exception as e:
            print(f"❌ Unexpected error in {step_key.upper()}: {e}")
            self.logger.error(f"Unexpected error in {step_key}: {e}")
            return False

    def print_summary(self, success, failed_steps):
        """Print execution summary"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 80)

        print(
            f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )

        if self.step_times:
            print("\n📈 Step execution times:")
            for step, duration in self.step_times.items():
                print(f"   {step}: {duration:.2f}s ({duration/60:.2f}m)")

        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("✅ All enabled steps executed without errors")
        else:
            print("\n⚠️  Pipeline completed with errors")
            if failed_steps:
                print(f"❌ Failed steps: {', '.join(failed_steps)}")

        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def run_pipeline(self, steps_override=None):
        """Run the complete pipeline"""
        self.start_time = time.time()
        self.print_banner()

        # Determine which steps to run
        if steps_override:
            steps_to_run = [f"step{i}" for i in steps_override]
        else:
            steps_to_run = [
                f"step{i}"
                for i in self.meta_config["pipeline_settings"]["steps_to_run"]
            ]

        success = True
        failed_steps = []

        print(
            f"\n🎯 Pipeline will execute {len(steps_to_run)} steps: {', '.join(steps_to_run)}"
        )

        # Execute each step
        for step_key in steps_to_run:
            if step_key not in self.meta_config["step_configurations"]:
                print(f"❌ Step configuration not found: {step_key}")
                failed_steps.append(step_key)
                success = False
                continue

            step_config = self.meta_config["step_configurations"][step_key]

            if not step_config.get("enabled", True):
                print(f"⏭️  Skipping disabled step: {step_key}")
                continue

            self.print_step_info(step_key, step_config)

            if not self.execute_step(step_key, step_config):
                failed_steps.append(step_key)
                success = False

                if not self.meta_config["pipeline_settings"]["continue_on_error"]:
                    break

        self.print_summary(success, failed_steps)
        return success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LIDAR Processing Pipeline Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all enabled steps
  %(prog)s --steps 1 2        # Run only steps 1 and 2
  %(prog)s --config custom.json # Use custom meta config
  %(prog)s --list             # List available steps
        """,
    )

    parser.add_argument(
        "--config",
        default="config/meta_config.json",
        help="Path to meta configuration file (default: config/meta_config.json)",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="Specific steps to run (e.g., --steps 1 2 3)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available steps and exit"
    )

    args = parser.parse_args()

    try:
        executor = PipelineExecutor(args.config)

        if args.list:
            print("Available pipeline steps:")
            for step_key, step_config in executor.meta_config[
                "step_configurations"
            ].items():
                status = (
                    "✅ Enabled" if step_config.get("enabled", True) else "❌ Disabled"
                )
                print(f"  {step_key}: {step_config['description']} ({status})")
            return

        success = executor.run_pipeline(args.steps)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
