import subprocess
import time
import sys
import os
import platform

# 設定要執行的檔案 (可以是 .py 或 .cpp)
SOLVER_FILE = 'solver - GPT5.2.cpp'
TIME_LIMIT_SECONDS = 120  

# 編譯 C++ 程式
def compile_cpp(cpp_file):
    exe_file = os.path.splitext(cpp_file)[0]
    if platform.system() == "Windows":
        exe_file += ".exe"
    
    print(f"Compiling {cpp_file}...")
    # 使用 g++ 編譯，開啟 O3 優化
    compile_cmd = ["g++", "-O3", cpp_file, "-o", exe_file]
    try:
        subprocess.check_call(compile_cmd)
        print("Compilation successful.")
        return exe_file
    except subprocess.CalledProcessError:
        print("Compilation failed.")
        return None

# 主程式
def main():
    if not os.path.exists(SOLVER_FILE):
        print(f"Error: {SOLVER_FILE} not found.")
        return

    cmd = []
    created_exe = None

    # 判斷檔案類型並準備指令
    if SOLVER_FILE.endswith('.py'):
        cmd = [sys.executable, SOLVER_FILE]
    elif SOLVER_FILE.endswith('.cpp'):
        created_exe = compile_cpp(SOLVER_FILE)
        if not created_exe:
            return
        # 使用絕對路徑確保能找到執行檔
        cmd = [os.path.abspath(created_exe)]
    else:
        print("Unsupported file type. Please use .py or .cpp")
        return

    print(f"開始執行 {SOLVER_FILE} 時間限制： {TIME_LIMIT_SECONDS} 秒...")
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"執行起始時間：{start_time_str}")
    print("---------------------------------------------------------")
    
    process = subprocess.Popen(cmd)
    
    end_time_str = ""
    try:
        # 等待 30 分鐘 (1800 秒)
        process.wait(timeout=TIME_LIMIT_SECONDS)
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"解題完成！！！執行結束時間：{end_time_str}")

    # 超過時間限制
    except subprocess.TimeoutExpired:
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"\n已達時間限制。執行結束時間：{end_time_str}")
        
        if platform.system() == "Windows":
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
        else:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
        print("強制終止解題程式。")
    
    finally:
        if not end_time_str:
            end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        try:
            with open("execution_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"Solver: {SOLVER_FILE}\n")
                log_file.write(f"Start: {start_time_str}\n")
                log_file.write(f"End:   {end_time_str}\n")
                log_file.write("-" * 40 + "\n")
            print(f"已將執行時間寫入 execution_log.txt")
        except Exception as e:
            print(f"寫入 log 失敗: {e}")

        # 清理編譯產生的 exe 檔 (如果是 C++ 模式)
        if created_exe and os.path.exists(created_exe):
            try:
                os.remove(created_exe)
                print(f"Cleaned up {created_exe}")
            except:
                pass

if __name__ == "__main__":
    main()
