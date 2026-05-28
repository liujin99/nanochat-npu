"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

# 新增：导入NPU支持
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

def run_command(cmd):
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # Return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except:
        return None

def get_git_info():
    """Get current git commit, branch, and dirty status."""
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # Get commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # First line, truncated

    return info

def get_gpu_info():
    """获取8卡NPU硬件信息"""
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        return {"available": False}

    num_devices = torch.npu.device_count()  # 8
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": [],
        "total_memory_gb": 0.0
    }

    for i in range(num_devices):
        props = torch.npu.get_device_properties(i)
        info["names"].append(props.name)
        mem_gb = props.total_memory / (1024**3)
        info["memory_gb"].append(mem_gb)
        info["total_memory_gb"] += mem_gb

    info["npu_version"] = torch_npu.__version__ if hasattr(torch_npu, '__version__') else "unknown"
    return info

def get_system_info():
    """Get system information."""
    info = {}

    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

# estimate_cost（适配 NPU 成本估算）
def estimate_cost(gpu_info, runtime_hours=None):
    """Estimate training cost based on GPU/NPU type and runtime."""

    # Rough pricing (适配NPU)
    default_rate = 2.0
    device_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
        "Ascend": 1.50,  # 昇腾NPU hourly rate
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify device type from name
    hourly_rate = None
    device_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for dev_type, rate in device_hourly_rates.items():
        if dev_type in device_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "device_type": device_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }


# generate_header（优化，适配 NPU + 展示）
def generate_header():
    """Generate the header for a training report (适配单卡/8卡NPU，完整最终版)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 采集各类信息
    git_info = get_git_info()
    device_info = get_gpu_info()  # 兼容NPU/GPU
    sys_info = get_system_info()
    cost_info = estimate_cost(device_info)

    # 确定设备类型标题（单卡/8卡NPU/GPU）
    device_title = "NPU" if "npu_version" in device_info else "GPU"
    card_count = device_info.get("count", 0)
    card_type = f"{card_count}卡" if card_count > 1 else "单卡"
    report_title = f"nanochat training report ({card_type} {device_title})"

    # 构建头部内容
    header = f"""# {report_title}

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Commit Message: {git_info['message'] or "No message"}

### Hardware
- Platform: {sys_info['platform']}
- Hostname: {sys_info['hostname']}
- CPUs: {sys_info['cpu_count']} physical cores ({sys_info['cpu_count_logical']} logical)
- System Memory: {sys_info['memory_gb']} GB
"""

    # 设备信息展示（区分NPU/GPU）
    if device_info.get("available"):
        dev_names = ", ".join(set(device_info["names"]))  # 去重设备名称
        header += f"""- Accelerators: {device_info['count']}x {dev_names}
- Total Accelerator Memory: {device_info['total_memory_gb']} GB
"""
        # 显示NPU/CUDA版本
        if "npu_version" in device_info:
            header += f"- NPU Version: {device_info['npu_version']}\n"
        if "cuda_version" in device_info:
            header += f"- CUDA Version: {device_info['cuda_version']}\n"
    else:
        header += "- Accelerators: None available (CPU only)\n"

    # 成本估算
    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate (estimated): ${cost_info['hourly_rate']}/hour\n"""

    # 软件信息
    header += f"""
### Software
- Python Version: {sys_info['python_version']}
- PyTorch Version: {sys_info['torch_version']}
- Working Directory: {sys_info['working_dir']}
- Output Directory: {sys_info['nanochat_base_dir']}

### Code & Dependencies (Bloat Metrics)
"""

    # 代码统计（兼容非Git仓库）
    extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
    git_patterns = ' '.join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or '').split('\n') if f.strip()]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0

    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            total_line = wc_output.strip().split('\n')[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                try:
                    num_lines = int(parts[0])
                    num_chars = int(parts[1])
                except ValueError:
                    num_lines = 0
                    num_chars = 0
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # 依赖统计（兼容uv.lock不存在）
    uv_lock_lines = 0
    uv_lock_path = os.path.join(sys_info['working_dir'], 'uv.lock')
    if os.path.exists(uv_lock_path):
        try:
            with open(uv_lock_path, 'r', encoding='utf-8') as f:
                uv_lock_lines = len(f.readlines())
        except Exception:
            uv_lock_lines = 0

    # 添加代码统计信息
    header += f"""
- Source Files (tracked): {num_files:,}
- Total Lines of Code: {num_lines:,}
- Total Characters: {num_chars:,}
- Estimated Tokens: {num_tokens:,}
- Dependency Lines (uv.lock): {uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """Slugify a text string."""
    return text.lower().replace(" ", "-")

# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "mid-model-training.md",      # 加入
    "mid-model-loss.md",        # 加入
    "mid-model-evaluation.md",    # 加入
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys] # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """Extract timestamp from content with given prefix."""
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {} # the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None # will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    end_time = extract_timestamp(section, "timestamp:")
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                # 添加 mid train
                if file_name == "mid-model-evaluation.md":
                    final_metrics["mid"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K") # RL only evals GSM8K
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # Fixed column widths
            stages = ["base", "mid", "sft", "rl"] # 记得修改
            metric_width = 15
            value_width = 8
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # also cp the report.md file to current directory
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """Reset the report."""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

class DummyReport:
    def log(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass

def get_report():
    # just for convenience, only rank 0 logs to report
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="Operation to perform (default: generate)")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()