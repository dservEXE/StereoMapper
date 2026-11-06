using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("Стереокамера — расстояние и карта")]
[assembly: AssemblyProduct("StereoMapper")]
[assembly: AssemblyVersion("0.14.1.0")]
[assembly: AssemblyFileVersion("0.14.1.0")]

internal static class Launcher
{
    // Windows CommandLineToArgvW / Python quoting, including a trailing slash.
    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') result.Append('\\', slashes * 2 + 1);
            else result.Append('\\', slashes);
            slashes = 0;
            result.Append(c);
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    [STAThread]
    private static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        bool checking = args.Length > 0 && args[0] == "--self-test";
        try
        {
            string python = Path.Combine(root, "runtime", "pythonw.exe");
            string entry = Path.Combine(root, "app_bootstrap.py");
            if (!File.Exists(python) || !File.Exists(entry))
                throw new IOException("Распакуйте архив целиком: рядом с программой должна быть папка runtime.");
            StringBuilder command = new StringBuilder("-I -X utf8 ");
            command.Append(Quote(entry));
            foreach (string arg in args) command.Append(" ").Append(Quote(arg));
            ProcessStartInfo info = new ProcessStartInfo(python, command.ToString());
            info.WorkingDirectory = root;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            using (Process child = Process.Start(info))
            {
                child.WaitForExit();
                if (child.ExitCode != 0 && !checking)
                    MessageBox.Show("Программа завершилась с ошибкой. Проверьте data\\logs\\startup.log в папке программы.",
                                    "Стереокамера", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            if (!checking) MessageBox.Show(error.Message, "Стереокамера", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
