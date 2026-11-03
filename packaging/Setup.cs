using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyProduct("StereoMapper Installer")]
[assembly: AssemblyVersion(BuildInfo.Version + ".0")]
internal static class Setup
{
    static readonly string[] Folders = { "stereo", "runtime", "models", "licenses" };
    static readonly string[] Files = { "StereoMapper.exe", "main.py", "app_bootstrap.py", "calibration_poses.png", "app.ico", "README.md", "SCANNING_GUIDE.md", "CALIBRATION_GUIDE.md", "3D_EDITOR_GUIDE.md", "INTERFACE_GUIDE.md", "START_HERE_RU.txt", "BUILD_INFO.json", "Install.cmd", "install.ps1" };
    static string Self { get { return Assembly.GetExecutingAssembly().Location; } }
    static string DefaultRoot { get { return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", BuildInfo.Folder); } }
    static string Q(string value) { return "\"" + value.Replace("\"", "") + "\""; }
    static bool Same(string a, string b) { return string.Equals(Path.GetFullPath(a).TrimEnd('\\'), Path.GetFullPath(b).TrimEnd('\\'), StringComparison.OrdinalIgnoreCase); }
    static void Ordinary(string file)
    {
        for (string p = Path.GetFullPath(file); !string.IsNullOrEmpty(p); p = Path.GetDirectoryName(p))
            if ((File.Exists(p) || Directory.Exists(p)) && (File.GetAttributes(p) & FileAttributes.ReparsePoint) != 0) throw new IOException("Путь содержит ссылку: " + p);
    }
    static void OrdinaryTree(string path)
    {
        Ordinary(path);
        if (Directory.Exists(path)) foreach (string item in Directory.EnumerateFileSystemEntries(path)) OrdinaryTree(item);
    }
    static long PayloadStart(FileStream input)
    {
        if (input.Length < 16) throw new IOException("Установщик загружен не полностью.");
        input.Position = input.Length - 16;
        byte[] footer = new byte[16];
        if (input.Read(footer, 0, 16) != 16 || Encoding.ASCII.GetString(footer, 0, 8) != "SMAPZIP1") throw new IOException("Установщик загружен не полностью.");
        long length = BitConverter.ToInt64(footer, 8);
        long start = input.Length - 16 - length;
        if (length <= 0 || start < 1024 || start > 16 * 1024 * 1024) throw new IOException("Некорректная длина пакета.");
        return start;
    }
    sealed class Slice : Stream
    {
        readonly Stream inner; readonly long start, length; long position;
        public Slice(Stream file, long offset, long size) { inner = file; start = offset; length = size; inner.Position = start; }
        public override bool CanRead { get { return true; } } public override bool CanWrite { get { return false; } } public override bool CanSeek { get { return true; } }
        public override long Length { get { return length; } } public override long Position { get { return position; } set { Seek(value, SeekOrigin.Begin); } }
        public override int Read(byte[] buffer, int offset, int count) { count = (int)Math.Min(count, length - position); if (count <= 0) return 0; inner.Position = start + position; int n = inner.Read(buffer, offset, count); position += n; return n; }
        public override long Seek(long offset, SeekOrigin origin) { long next = origin == SeekOrigin.Begin ? offset : origin == SeekOrigin.Current ? position + offset : length + offset; if (next < 0 || next > length) throw new IOException("Недопустимое смещение пакета."); position = next; inner.Position = start + next; return next; }
        public override void Flush() { } public override void SetLength(long value) { throw new NotSupportedException(); } public override void Write(byte[] b, int o, int c) { throw new NotSupportedException(); }
        protected override void Dispose(bool disposing) { if (disposing) inner.Dispose(); base.Dispose(disposing); }
    }
    static Stream Payload()
    {
        var input = File.OpenRead(Self);
        try { long start = PayloadStart(input); return new Slice(input, start, input.Length - start - 16); } catch { input.Dispose(); throw; }
    }
    static bool Allowed(string relative)
    {
        string top = relative.Split('\\')[0];
        if (BuildInfo.CodeOnly) return string.Equals(top, "stereo", StringComparison.OrdinalIgnoreCase);
        foreach (string folder in Folders) if (string.Equals(folder, top, StringComparison.OrdinalIgnoreCase)) return true;
        foreach (string file in Files) if (string.Equals(file, relative, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }
    static void VerifyPayload()
    {
        using (var input = Payload()) using (var sha = SHA256.Create())
            if (BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "") != BuildInfo.Hash) throw new IOException("Проверка SHA-256 не пройдена.");
        using (var input = Payload()) using (var zip = new ZipArchive(input, ZipArchiveMode.Read))
            if (zip.Entries.Count == 0) throw new IOException("Пакет пуст.");
    }
    static void Extract(string destination, Action<int, string> progress)
    {
        Ordinary(destination);
        if (Directory.Exists(destination) && Directory.EnumerateFileSystemEntries(destination).GetEnumerator().MoveNext()) throw new IOException("Папка подготовки не пуста.");
        Directory.CreateDirectory(destination);
        progress(0, "Проверяю целостность установщика…");
        using (var input = Payload()) using (var sha = SHA256.Create())
            if (BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "") != BuildInfo.Hash) throw new IOException("Проверка SHA-256 не пройдена. Скачайте установщик заново.");
        string prefix = Path.GetFullPath(destination).TrimEnd('\\') + "\\";
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        using (var input = Payload()) using (var archive = new ZipArchive(input, ZipArchiveMode.Read))
        {
            bool wrapped = archive.Entries.Count > 0 && archive.Entries[0].FullName.Replace('/', '\\').StartsWith("StereoMapper\\", StringComparison.OrdinalIgnoreCase);
            long total = 0;
            foreach (var e in archive.Entries) { total = checked(total + e.Length); }
            if (total <= 0 || total > (BuildInfo.CodeOnly ? 64L * 1024 * 1024 : 12L * 1024 * 1024 * 1024)) throw new IOException("Недопустимый размер содержимого.");
            long done = 0; int lastPercent = -1;
            foreach (var entry in archive.Entries)
            {
                string relative = entry.FullName.Replace('/', '\\');
                if (wrapped) { if (!relative.StartsWith("StereoMapper\\", StringComparison.OrdinalIgnoreCase)) throw new IOException("Смешанные корни пакета."); relative = relative.Substring(13); }
                if (relative.Length == 0 && entry.Length == 0) continue;
                if (relative.IndexOf(':') >= 0 || Path.IsPathRooted(relative) || ((entry.ExternalAttributes >> 16) & 0xf000) == 0xa000 || (entry.ExternalAttributes & 0x400) != 0) throw new IOException("Недопустимый путь в пакете.");
                string target = Path.GetFullPath(Path.Combine(destination, relative));
                if (!target.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || !Allowed(target.Substring(prefix.Length).TrimEnd('\\')) || !names.Add(target)) throw new IOException("Пакет содержит лишние файлы: " + relative);
                if (relative.EndsWith("\\")) { Directory.CreateDirectory(target); continue; }
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                entry.ExtractToFile(target, false);
                done += entry.Length;
                int percent = (int)(100 * done / total);
                if (percent != lastPercent) { lastPercent = percent; progress(5 + 75 * percent / 100, "Распаковываю программу… " + percent + "%"); }
            }
        }
        foreach (string name in BuildInfo.CodeOnly ? new[] { "stereo\\app.py", "stereo\\version.py" } : new[] { "StereoMapper.exe", "main.py", "app_bootstrap.py", "stereo\\app.py", "stereo\\version.py", "runtime\\python.exe", "runtime\\pythonw.exe", "runtime\\Lib\\os.py", "runtime\\Lib\\site-packages\\cv2\\__init__.py", "runtime\\Lib\\site-packages\\numpy\\__init__.py", "runtime\\Lib\\site-packages\\PIL\\__init__.py" })
            if (!File.Exists(Path.Combine(destination, name))) throw new IOException("Пакет приложения неполон: " + name);
    }
    static void DeleteOwned(string directory, string parent, string marker)
    {
        if (string.IsNullOrEmpty(directory) || !Same(Path.GetDirectoryName(directory), parent) || !Path.GetFileName(directory).StartsWith(marker, StringComparison.Ordinal)) return;
        try { if (Directory.Exists(directory)) { OrdinaryTree(directory); Directory.Delete(directory, true); } } catch { }
    }
    static void Move(string from, string to) { if (Directory.Exists(from)) Directory.Move(from, to); else File.Move(from, to); }
    static void WaitForProcesses(string root, bool updating)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(updating ? 30 : 0);
        while (true)
        {
            bool active = false;
            foreach (var process in Process.GetProcesses())
            {
                try { if (process.Id != Process.GetCurrentProcess().Id && process.MainModule.FileName.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase)) active = true; }
                catch (Win32Exception) { } catch (InvalidOperationException) { } finally { process.Dispose(); }
            }
            if (!active) return;
            if (DateTime.UtcNow >= deadline) throw new IOException("Сначала полностью закройте StereoMapper.");
            Thread.Sleep(100);
        }
    }
    static void WaitForParent(int pid, string expected)
    {
        if (pid <= 0) return;
        try { using (var process = Process.GetProcessById(pid)) { if (!Same(process.MainModule.FileName, expected)) throw new IOException("Процесс приложения изменился."); if (!process.WaitForExit(30000)) throw new IOException("Приложение не завершилось. Установка отменена."); } } catch (ArgumentException) { }
    }
    static void Shortcut(string root)
    {
        var shell = Activator.CreateInstance(Type.GetTypeFromProgID("WScript.Shell"));
        dynamic link = shell.GetType().InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "StereoMapper.lnk") });
        link.TargetPath = Path.Combine(root, BuildInfo.Executable); link.WorkingDirectory = root; link.IconLocation = link.TargetPath + ",0"; link.Save();
    }
    static void CopyExecutable(string path)
    {
        using (var input = File.OpenRead(Self)) using (var output = File.Create(path))
        {
            long length; try { length = PayloadStart(input); } catch (IOException) { length = input.Length; }
            input.Position = 0; byte[] buffer = new byte[65536];
            while (length > 0) { int n = input.Read(buffer, 0, (int)Math.Min(buffer.Length, length)); if (n == 0) throw new EndOfStreamException(); output.Write(buffer, 0, n); length -= n; }
        }
    }
    static void Register(string root)
    {
        string uninstaller = Path.Combine(root, "Uninstall.exe"); CopyExecutable(uninstaller);
        using (var key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\StereoMapper"))
        {
            key.SetValue("DisplayName", "StereoMapper"); key.SetValue("DisplayVersion", BuildInfo.Version); key.SetValue("Publisher", "StereoMapper"); key.SetValue("InstallLocation", root);
            key.SetValue("DisplayIcon", Path.Combine(root, BuildInfo.Executable)); key.SetValue("UninstallString", Q(uninstaller) + " --uninstall " + Q(root)); key.SetValue("NoModify", 1); key.SetValue("NoRepair", 1);
        }
    }
    static void Install(string chosenRoot, int pid, string parentPath, string cleanup, bool launch, bool shortcut, bool register, Action<int, string> progress)
    {
        string root = Path.GetFullPath(chosenRoot).TrimEnd('\\'); Ordinary(root);
        if (Path.GetPathRoot(root).TrimEnd('\\').Equals(root, StringComparison.OrdinalIgnoreCase)) throw new IOException("Выберите отдельную папку программы.");
        if (BuildInfo.CodeOnly && (!File.Exists(Path.Combine(root, "runtime\\pythonw.exe")) || !File.Exists(Path.Combine(root, BuildInfo.Executable)) || !File.Exists(Path.Combine(root, "stereo\\app.py")))) throw new IOException("Для первой установки нужен StereoMapper-Setup.exe. Обновление применяется к готовому приложению с runtime.");
        if (Directory.Exists(root) && Directory.EnumerateFileSystemEntries(root).GetEnumerator().MoveNext() && !File.Exists(Path.Combine(root, BuildInfo.Executable))) throw new IOException("Папка занята другими файлами. Выберите другую папку.");
        string parent = Path.GetDirectoryName(root); Directory.CreateDirectory(parent);
        string token = Guid.NewGuid().ToString("N"), stage = Path.Combine(parent, ".StereoMapper-stage-" + token), backup = Path.Combine(parent, ".StereoMapper-previous-" + token);
        var movedOld = new List<string>(); var movedNew = new List<string>();
        using (var mutex = new Mutex(false, "Local\\StereoMapper-Installer"))
        {
            bool acquired; try { acquired = mutex.WaitOne(0); } catch (AbandonedMutexException) { acquired = true; }
            if (!acquired) throw new IOException("Другая установка уже выполняется.");
            try
            {
                if (pid <= 0) WaitForProcesses(root, false);
                Extract(stage, progress);
                WaitForParent(pid, parentPath); WaitForProcesses(root, pid > 0);
                Directory.CreateDirectory(root); Directory.CreateDirectory(backup);
                foreach (string staged in Directory.EnumerateFileSystemEntries(stage)) OrdinaryTree(Path.Combine(root, Path.GetFileName(staged)));
                progress(85, "Устанавливаю файлы. Данные и калибровка сохраняются…");
                foreach (string staged in Directory.GetFileSystemEntries(stage))
                {
                    string name = Path.GetFileName(staged), target = Path.Combine(root, name), old = Path.Combine(backup, name);
                    if (File.Exists(target) || Directory.Exists(target)) { Move(target, old); movedOld.Add(name); }
                    Move(staged, target); movedNew.Add(name);
                }
                // Integration failures do not invalidate the installed, complete runtime.
                if (shortcut) { try { Shortcut(root); } catch { } }
                if (register) { try { Register(root); } catch { } }
            }
            catch
            {
                foreach (string name in movedNew) { string target = Path.Combine(root, name); try { if (Directory.Exists(target)) { OrdinaryTree(target); Directory.Delete(target, true); } else if (File.Exists(target)) File.Delete(target); } catch { } }
                foreach (string name in movedOld) { string old = Path.Combine(backup, name), target = Path.Combine(root, name); if (!File.Exists(target) && !Directory.Exists(target)) Move(old, target); }
                throw;
            }
            finally { DeleteOwned(stage, parent, ".StereoMapper-stage-"); mutex.ReleaseMutex(); }
        }
        DeleteOwned(backup, parent, ".StereoMapper-previous-");
        progress(100, "Готово. StereoMapper установлен.");
        if (launch)
        {
            string args = cleanup == null ? "" : "--cleanup-update " + Q(cleanup) + " " + Process.GetCurrentProcess().Id;
            Process.Start(new ProcessStartInfo(Path.Combine(root, BuildInfo.Executable), args) { UseShellExecute = false, WorkingDirectory = root });
        }
    }
    static void Uninstall(string root)
    {
        root = Path.GetFullPath(root).TrimEnd('\\'); OrdinaryTree(root);
        if (!File.Exists(Path.Combine(root, BuildInfo.Executable)) || !File.Exists(Path.Combine(root, "main.py"))) throw new IOException("Папка StereoMapper не найдена.");
        WaitForProcesses(root, false);
        foreach (string folder in Folders) { string target = Path.Combine(root, folder); if (Directory.Exists(target)) Directory.Delete(target, true); }
        foreach (string file in Files) { string target = Path.Combine(root, file); if (File.Exists(target)) File.Delete(target); }
        string uninstaller = Path.Combine(root, "Uninstall.exe"); if (File.Exists(uninstaller)) File.Delete(uninstaller);
        using (var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\StereoMapper"))
            if (key != null && Same(Convert.ToString(key.GetValue("InstallLocation")), root)) Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\StereoMapper", false);
        string link = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "StereoMapper.lnk");
        if (File.Exists(link))
        {
            dynamic shell = Activator.CreateInstance(Type.GetTypeFromProgID("WScript.Shell")); dynamic shortcut = shell.CreateShortcut(link);
            if (Same(Convert.ToString(shortcut.TargetPath), Path.Combine(root, BuildInfo.Executable))) File.Delete(link);
        }
        // data/, output/, settings.json and screen_board.json deliberately remain.
    }
    static string Argument(string[] args, string name, string fallback) { int i = Array.IndexOf(args, name); return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback; }
    static bool Has(string[] args, string name) { return Array.IndexOf(args, name) >= 0; }
    static void StatusFile(string[] args, Exception error)
    {
        string file = Argument(args, "--result", null); if (file == null) return;
        File.WriteAllText(file, error == null ? "OK" : error.ToString(), Encoding.UTF8);
    }
    [STAThread] static int Main(string[] args)
    {
        if (Has(args, "--verify") || Has(args, "--verify-payload") || Has(args, "--install"))
        {
            try
            {
                if (Has(args, "--verify-payload")) VerifyPayload();
                else if (Has(args, "--verify")) Extract(Argument(args, "--verify", null), delegate { });
                else Install(Argument(args, "--directory", DefaultRoot), 0, null, null, !Has(args, "--no-launch"), !Has(args, "--no-shortcut"), !Has(args, "--no-registry"), delegate { });
                StatusFile(args, null); return 0;
            }
            catch (Exception error) { StatusFile(args, error); return 1; }
        }
        Application.EnableVisualStyles();
        if (args.Length == 2 && args[0] == "--uninstall")
        {
            try
            {
                if (MessageBox.Show("Удалить программу StereoMapper? Исследования, калибровка и настройки останутся в её папке.", "StereoMapper", MessageBoxButtons.YesNo) != DialogResult.Yes) return 0;
                string temporary = Path.Combine(Path.GetTempPath(), "StereoMapper-uninstall-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(temporary);
                string helper = Path.Combine(temporary, "Uninstall.exe"); CopyExecutable(helper);
                Process.Start(new ProcessStartInfo(helper, "--remove " + Q(Path.GetFullPath(args[1])) + " " + Process.GetCurrentProcess().Id) { UseShellExecute = false }); return 0;
            }
            catch (Exception error) { MessageBox.Show(error.Message, "StereoMapper"); return 1; }
        }
        if (args.Length == 3 && args[0] == "--remove")
        {
            try { using (var p = Process.GetProcessById(int.Parse(args[2]))) p.WaitForExit(30000); } catch (ArgumentException) { }
            try { Uninstall(args[1]); MessageBox.Show("Программа удалена. Исследования и настройки сохранены.", "StereoMapper"); return 0; } catch (Exception error) { MessageBox.Show(error.Message, "StereoMapper"); return 1; }
        }
        if (args.Length == 5 && args[0] == "--update")
        {
            try
            {
                int pid; if (!int.TryParse(args[1], out pid) || pid <= 0) throw new IOException("Неверный процесс приложения.");
                string cleanup = Path.GetFullPath(args[4]).TrimEnd('\\'); Ordinary(cleanup);
                string updates = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "StereoMapper", "updates"); Guid token;
                if (!Same(Path.GetDirectoryName(cleanup), updates) || !Guid.TryParseExact(Path.GetFileName(cleanup), "N", out token) || !Same(Self, Path.Combine(cleanup, BuildInfo.CodeOnly ? "StereoMapper-Update.exe" : "StereoMapper-Setup.exe"))) throw new IOException("Не удалось проверить источник обновления.");
                string source = Path.GetFullPath(args[2]), chosen = Path.GetFullPath(args[3]);
                if (!Same(source, Path.Combine(chosen, "runtime\\pythonw.exe"))) throw new IOException("Не удалось проверить приложение.");
                Install(chosen, pid, source, cleanup, true, true, true, delegate { }); return 0;
            }
            catch (Exception error) { MessageBox.Show("Обновление не установлено: " + error.Message + "\nДанные и настройки сохранены.", "StereoMapper"); return 1; }
        }
        using (var window = new Form { Text = "StereoMapper " + BuildInfo.Version, ClientSize = new Size(640, 340), StartPosition = FormStartPosition.CenterScreen, FormBorderStyle = FormBorderStyle.FixedDialog, MaximizeBox = false, Font = new Font("Segoe UI", 10) })
        {
            var title = new Label { Text = BuildInfo.CodeOnly ? "Обновление StereoMapper" : "Установка StereoMapper", Font = new Font("Segoe UI", 17, FontStyle.Bold), Location = new Point(24, 18), Size = new Size(590, 35) };
            var label = new Label { Text = BuildInfo.CodeOnly ? "Новая версия программы. Снимки, карты, калибровка, Python и модели сохраняются." : "Python и OpenCV уже включены. Права администратора не нужны. Снимки и настройки из прежних испытаний не включены.", Location = new Point(24, 65), Size = new Size(590, 55) };
            var folder = new TextBox { Text = DefaultRoot, Location = new Point(24, 128), Width = 485 };
            var browse = new Button { Text = "Папка…", Location = new Point(518, 126), Width = 95 };
            browse.Click += delegate { using (var dialog = new FolderBrowserDialog { SelectedPath = folder.Text }) if (dialog.ShowDialog() == DialogResult.OK) folder.Text = dialog.SelectedPath; };
            var launch = new CheckBox { Text = "Запустить программу после установки", Checked = true, Location = new Point(24, 167), Width = 590 };
            var status = new Label { Text = "Выберите папку и нажмите «Установить».", Location = new Point(24, 202), Size = new Size(590, 27) };
            var bar = new ProgressBar { Location = new Point(24, 233), Size = new Size(590, 15) };
            var install = new Button { Text = "Установить " + BuildInfo.Version, Location = new Point(24, 270), Size = new Size(590, 45) };
            bool working = false;
            window.FormClosing += delegate(object sender, FormClosingEventArgs e) { if (working) e.Cancel = true; };
            install.Click += delegate
            {
                string target = folder.Text; bool run = launch.Checked;
                working = true; install.Enabled = browse.Enabled = folder.Enabled = launch.Enabled = false;
                var worker = new BackgroundWorker();
                worker.DoWork += delegate { Install(target, 0, null, null, run, true, true, delegate(int percent, string text) { window.BeginInvoke((Action)delegate { bar.Value = percent; status.Text = text; }); }); };
                worker.RunWorkerCompleted += delegate(object sender, RunWorkerCompletedEventArgs e)
                {
                    working = false;
                    if (e.Error == null) window.Close();
                    else { MessageBox.Show(e.Error.Message, "StereoMapper", MessageBoxButtons.OK, MessageBoxIcon.Error); install.Enabled = browse.Enabled = folder.Enabled = launch.Enabled = true; status.Text = "Установка не завершена. Данные сохранены."; }
                    worker.Dispose();
                };
                worker.RunWorkerAsync();
            };
            window.Controls.AddRange(new Control[] { title, label, folder, browse, launch, status, bar, install }); Application.Run(window);
        }
        return 0;
    }
}
