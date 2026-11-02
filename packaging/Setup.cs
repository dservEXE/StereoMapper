using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Windows.Forms;
[assembly: AssemblyProduct(BuildInfo.Name + " Installer")]
[assembly: AssemblyVersion(BuildInfo.Version + ".0")]
internal static class Setup
{
    static string Q(string value) { return "\"" + value.Replace("\"", "") + "\""; }
    static string DefaultRoot { get { return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", BuildInfo.Folder); } }
    static string Self { get { return Assembly.GetExecutingAssembly().Location; } }
    static void Ordinary(string file)
    {
        for (var current = Path.GetFullPath(file); !string.IsNullOrEmpty(current); current = Path.GetDirectoryName(current))
            if ((File.Exists(current) || Directory.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0) throw new IOException("Путь содержит ссылку: " + current);
    }
    static string Extract(string destination)
    {
        Directory.CreateDirectory(destination);
        var zip = Path.Combine(destination, ".payload.zip");
        using (var input = Assembly.GetExecutingAssembly().GetManifestResourceStream("Payload"))
        using (var output = File.Create(zip)) { if (input == null) throw new IOException("Missing payload"); input.CopyTo(output); }
        using (var input = File.OpenRead(zip)) using (var sha = SHA256.Create())
            if (BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "") != BuildInfo.Hash) throw new IOException("Проверка пакета не прошла.");
        var root = Path.GetFullPath(destination).TrimEnd('\\') + "\\";
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        using (var archive = ZipFile.OpenRead(zip))
        {
            foreach (var entry in archive.Entries)
            {
                var relative = entry.FullName.Replace('/', '\\');
                if (relative.IndexOf(':') >= 0 || Path.IsPathRooted(relative) || ((entry.ExternalAttributes >> 16) & 0xf000) == 0xa000) throw new IOException("Недопустимый путь в пакете.");
                var target = Path.GetFullPath(Path.Combine(destination, relative));
                if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase) || !names.Add(target)) throw new IOException("Недопустимое содержимое пакета.");
                if (BuildInfo.CodeOnly && !relative.StartsWith("stereo\\", StringComparison.OrdinalIgnoreCase)) throw new IOException("Пакет кода содержит лишние файлы.");
                if (relative.EndsWith("\\")) { Directory.CreateDirectory(target); continue; }
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                entry.ExtractToFile(target, false);
            }
        }
        File.Delete(zip);
        if (!File.Exists(Path.Combine(destination, BuildInfo.CodeOnly ? "stereo\\version.py" : BuildInfo.Executable))) throw new IOException("Пакет приложения неполон.");
        return destination;
    }
    static void DeleteOwned(string directory) { try { if (Directory.Exists(directory)) { Ordinary(directory); Directory.Delete(directory, true); } } catch { } }
    static void WaitForProcesses(string root, bool updating)
    {
        var deadline = DateTime.UtcNow.AddSeconds(updating ? 10 : 0);
        while (true)
        {
            bool active = false;
            foreach (var process in Process.GetProcesses())
            {
                try { if (process.Id != Process.GetCurrentProcess().Id && process.MainModule.FileName.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase)) active = true; }
                catch (System.ComponentModel.Win32Exception) { } catch (InvalidOperationException) { }
                finally { process.Dispose(); }
            }
            if (!active) return;
            if (DateTime.UtcNow >= deadline) throw new IOException("Сначала полностью закройте " + BuildInfo.Name + ".");
            System.Threading.Thread.Sleep(100);
        }
    }
    static void Shortcut(string root)
    {
        var shell = Activator.CreateInstance(Type.GetTypeFromProgID("WScript.Shell"));
        dynamic link = shell.GetType().InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), BuildInfo.Name + ".lnk") });
        link.TargetPath = Path.Combine(root, BuildInfo.Executable); link.WorkingDirectory = root; link.IconLocation = link.TargetPath + ",0"; link.Save();
    }
    static void Install(string chosenRoot, int pid, string parentPath, string cleanup)
    {
        var root = Path.GetFullPath(BuildInfo.CodeOnly ? chosenRoot : DefaultRoot).TrimEnd('\\'); Ordinary(root);
        if (BuildInfo.CodeOnly && (!File.Exists(Path.Combine(root, "runtime\\pythonw.exe")) || !File.Exists(Path.Combine(root, BuildInfo.Executable)) || !File.Exists(Path.Combine(root, "stereo\\app.py")))) throw new IOException("Выберите папку готового StereoMapper с runtime и StereoMapper.exe. Данные и калибровка сохранятся.");
        if (!BuildInfo.CodeOnly && Directory.Exists(root) && !File.Exists(Path.Combine(root, BuildInfo.Executable))) throw new IOException("Папка установки занята другими файлами.");
        var destination = BuildInfo.CodeOnly ? Path.Combine(root, "stereo") : root;
        var parent = Path.GetDirectoryName(destination); Directory.CreateDirectory(parent);
        var stage = Path.Combine(parent, "." + BuildInfo.Name + "-stage-" + Guid.NewGuid().ToString("N"));
        var backup = Path.Combine(parent, "." + BuildInfo.Name + "-previous-" + Guid.NewGuid().ToString("N"));
        bool oldMoved = false, newMoved = false, launched = false;
        using (var mutex = new System.Threading.Mutex(false, "Local\\" + BuildInfo.Name + "-Installer"))
        {
            bool acquired; try { acquired = mutex.WaitOne(0); } catch (System.Threading.AbandonedMutexException) { acquired = true; }
            if (!acquired) throw new IOException("Другая установка уже выполняется.");
            try
            {
                Extract(stage);
                if (pid > 0)
                {
                    try { using (var process = Process.GetProcessById(pid)) { if (!string.Equals(Path.GetFullPath(process.MainModule.FileName), Path.GetFullPath(parentPath), StringComparison.OrdinalIgnoreCase)) throw new IOException("Процесс приложения изменился."); if (!process.WaitForExit(30000)) throw new IOException("Приложение не завершилось. Установка отменена."); } } catch (ArgumentException) { }
                }
                WaitForProcesses(root, pid > 0);
                if (Directory.Exists(destination)) { Directory.Move(destination, backup); oldMoved = true; }
                Directory.Move(BuildInfo.CodeOnly ? Path.Combine(stage, "stereo") : stage, destination); newMoved = true;
                Shortcut(root);
                var args = cleanup == null ? "" : "--cleanup-update " + Q(cleanup) + " " + Process.GetCurrentProcess().Id;
                Process.Start(new ProcessStartInfo(Path.Combine(root, BuildInfo.Executable), args) { UseShellExecute = false, WorkingDirectory = root }); launched = true;
                DeleteOwned(backup);
            }
            catch
            {
                if (!launched) { if (newMoved) DeleteOwned(destination); if (oldMoved && !Directory.Exists(destination)) Directory.Move(backup, destination); }
                throw;
            }
            finally { DeleteOwned(stage); mutex.ReleaseMutex(); }
        }
    }
    [STAThread] static int Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--verify") { try { Extract(args[1]); return 0; } catch (Exception error) { Console.Error.WriteLine(error); return 1; } }
        Application.EnableVisualStyles();
        if (args.Length == 5 && args[0] == "--update")
        {
            try
            {
                int pid; if (!int.TryParse(args[1], out pid) || pid <= 0) throw new IOException("Неверный процесс приложения.");
                var cleanup = Path.GetFullPath(args[4]).TrimEnd('\\'); Ordinary(cleanup);
                var updates = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), BuildInfo.Name, "updates");
                Guid token;
                if (!string.Equals(Path.GetDirectoryName(cleanup), updates, StringComparison.OrdinalIgnoreCase) || !Guid.TryParseExact(Path.GetFileName(cleanup), "N", out token) || !string.Equals(Self, Path.Combine(cleanup, BuildInfo.Name + "-Setup.exe"), StringComparison.OrdinalIgnoreCase)) throw new IOException("Не удалось проверить источник обновления.");
                var source = Path.GetFullPath(args[2]); var chosen = Path.GetFullPath(args[3]);
                var expected = Path.Combine(chosen, BuildInfo.CodeOnly ? "runtime\\pythonw.exe" : BuildInfo.Executable);
                if (!string.Equals(source, expected, StringComparison.OrdinalIgnoreCase)) throw new IOException("Не удалось проверить приложение.");
                Install(chosen, pid, source, cleanup); return 0;
            }
            catch (Exception error) { MessageBox.Show("Обновление не установлено: " + error.Message + "\nДанные и настройки сохранены.", BuildInfo.Name); return 1; }
        }
        using (var window = new Form { Text = BuildInfo.Name + " " + BuildInfo.Version, ClientSize = new Size(610, 230), StartPosition = FormStartPosition.CenterScreen, FormBorderStyle = FormBorderStyle.FixedDialog, MaximizeBox = false, Font = new Font("Segoe UI", 10) })
        {
            var label = new Label { Text = BuildInfo.CodeOnly ? "Обновление готового StereoMapper.\nСреда Python, модели, снимки и калибровка сохраняются." : "Установка Kadr для текущего пользователя.\nПрофиль и сеанс источника сохраняются. Ярлык будет на рабочем столе.", Location = new Point(24, 20), Size = new Size(560, 60) };
            var folder = new TextBox { Text = DefaultRoot, Location = new Point(24, 95), Width = 450, ReadOnly = !BuildInfo.CodeOnly };
            var browse = new Button { Text = "Папка…", Location = new Point(484, 92), Width = 100, Enabled = BuildInfo.CodeOnly };
            browse.Click += delegate { using (var dialog = new FolderBrowserDialog { SelectedPath = folder.Text }) if (dialog.ShowDialog() == DialogResult.OK) folder.Text = dialog.SelectedPath; };
            var install = new Button { Text = "Установить " + BuildInfo.Version, Location = new Point(24, 150), Size = new Size(560, 45) };
            install.Click += delegate { install.Enabled = false; try { Install(folder.Text, 0, null, null); window.Close(); } catch (Exception error) { MessageBox.Show(error.Message, BuildInfo.Name); install.Enabled = true; } };
            window.Controls.AddRange(new Control[] { label, folder, browse, install }); Application.Run(window);
        }
        return 0;
    }
}
