using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.ComponentModel;
using System.Management;

namespace PythonServers
{

    class Program
    {
        // common to both servers
        static Char drive = 'D';

        // Entry rules server
        static int PORT = 50148; // 0; // https://stackoverflow.com/questions/138043/find-the-next-tcp-port-in-net
        static string HOST = "127.0.0.1";
        static System.Diagnostics.Process residentPython; // python.exe
        static string scriptName = "EntryDirectionPredictionTCPv3.py";
        static string modelsFolderName = "EntryModels_v3";

        static string modelsFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\"+ modelsFolderName; // drive+":\\EntryModels_v3";
        static string dummyJSONname = "several.13barJSONsamples.txt";

        // Exit rules server
        static int PORTexit = 50149; // 0; // https://stackoverflow.com/questions/138043/find-the-next-tcp-port-in-net
        static string HOSTexit = "127.0.0.1";
        static System.Diagnostics.Process residentPythonExit; // python.exe
        static string scriptNameExit = "TickDirectionPredictionTCP.py";
        static string modelsFolderNameExit = "fitted_models";
        static string modelsFolderExit = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments) + "\\" + modelsFolderNameExit;  // drive+":\\fitted_models";
        static string dummyJSONnameExit = "dummyExitData.json";
        static string columnsPath = modelsFolderExit + "\\" + "columns.json";

        // Start with Minimized window
        // https://stackoverflow.com/questions/44675085/minimize-console-at-app-startup-c-sharp
        [DllImport("User32.dll", CallingConvention = CallingConvention.StdCall, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool ShowWindow([In] IntPtr hWnd, [In] int nCmdShow);

        static string GetPythonPath()
        {
            //						Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
            //						Type "help", "copyright", "credits" or "license" for more information.
            //						>>>					
            try
            {
                string result = "python";
                //						using (Process myProcess = new Process())
                //						{
                Process myProcess = new Process();
                myProcess.StartInfo.UseShellExecute = true; //false; //true;

                // When UseShellExecute is true, WorkingDirectory gets or sets the directory that contains the process to be started.
                // When UseShellExecute is true, the working directory of the [NT8] application that starts the executable is also the working directory of the executable.
                // the default working directory is %SYSTEMROOT%\system32
                // the fully qualified name of the directory that contains the process to be started. 

                // You can start any process, HelloWorld is a do-nothing example.
                myProcess.StartInfo.FileName = "python";
                myProcess.StartInfo.CreateNoWindow = true;

                //							myProcess.StartInfo.RedirectStandardError = true;
                //							myProcess.StartInfo.RedirectStandardOutput = true;

                myProcess.StartInfo.Arguments = @"--version";

                // To avoid deadocks use an asynchronous read operation on at least one of the streams
                //string eOut = null;
                //						myProcess.ErrorDataReceived += new DataReceivedEventHandler((sender ,e) => {eOut += e.Data; });
                //string resultOut = null;
                //						myProcess.OutputDataReceived += new DataReceivedEventHandler((sender ,e) => {resultOut += e.Data; });

                ////							Print(this.Name+":GetPythonPath(): Python WorkingDirectory:"+ myProcess.StartInfo.WorkingDirectory);
                //							NinjaTrader.Code.Output.Process(this.Name+":70 GetPythonPath(): Python WorkingDirectory: '"+ myProcess.StartInfo.WorkingDirectory+"'",PrintTo.OutputTab1);
                //							NinjaTrader.Code.Output.Process(this.Name+":70 GetPythonPath(): Python Arguments: '"+ myProcess.StartInfo.Arguments+"'",PrintTo.OutputTab1);


                int msToWait = 10000; // max. 10 sec
                using (Process process = Process.Start(myProcess.StartInfo))
                {
                    process.Start();

                    DateTime dt0 = DateTime.Now;
                    if (process.MainModule == null) // GetPythonPath(): System.ComponentModel.Win32Exception (0x80004005): A 32 bit processes cannot access modules of a 64 bit process.
                    {
                        while ((DateTime.Now - dt0).TotalMilliseconds < msToWait)
                        {
                            string s = "";
                            try
                            {
                                s = process.MainModule.FileName;
                                break;
                            }
                            catch (Exception e) { };

                        } //while	
                    }

                    else  // process.MainModule != null :

                    //if (process.MainModule.FileName == null )
                    {
                        while ((DateTime.Now - dt0).TotalMilliseconds < msToWait)
                        {
                            string s = "";
                            try
                            {
                                s = process.MainModule.FileName;
                                break;
                            }
                            catch (Exception e) { };

                        } //while	
                    }


                    try
                    {
                        result = process.MainModule.FileName;
                        Console.WriteLine("GetPythonPath(): Python WorkingDirectory: '" + process.StartInfo.WorkingDirectory + "'");
                        Console.WriteLine(
                        "ID:" + process.Id + "\n" +
                        "MachineName:" + process.MachineName + "\n" +
                        "ProcessName:" + process.ProcessName + "\n" +
                        "path:" + process.MainModule.FileName + "\n" +
                        " retrieved in " + (DateTime.Now - dt0).TotalMilliseconds + "ms");
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine("GetPythonPath(): python.exe did not launch in " + ((DateTime.Now - dt0).TotalMilliseconds / 1000) + "sec ...." + e);
                        //return result;
                    }


                    //result = process.MainModule.FileName;

                    process.Close(); // kill all subprocesses of Python, if any
                                     //							process.Kill(); // kill all subprocesses of Python, if any
                    return result;
                } // using
            }// try
            catch (Exception e1)
            {
                Console.WriteLine("GetPythonPath(): " + e1);

            }
            return "";
        } // GetPythonPath()

        // https://learn.microsoft.com/en-us/dotnet/standard/io/how-to-copy-directories?redirectedfrom=MSDN
        static bool CopyDirectory(string sourceDir, string destinationDir, bool recursive)
        {
            // Get information about the source directory
            var dir = new DirectoryInfo(sourceDir);

            // Check if the source directory exists
            if (!dir.Exists)
            {
                //				throw new DirectoryNotFoundException($"Source directory not found: {dir.FullName}");
                //Print(this.Name+ (new DirectoryNotFoundException($"Source directory not found: {dir.FullName}")));
                Console.WriteLine($"Source directory not found: " + dir.FullName );
                return false;
            }

            // Cache directories before we start copying
            DirectoryInfo[] dirs = dir.GetDirectories();

            // Create the destination directory
            Directory.CreateDirectory(destinationDir);

            // Get the files in the source directory and copy to the destination directory
            foreach (FileInfo file in dir.GetFiles())
            {
                string targetFilePath = Path.Combine(destinationDir, file.Name);
                file.CopyTo(targetFilePath);
            }

            // If recursive and copying subdirectories, recursively call this method
            if (recursive)
            {
                foreach (DirectoryInfo subDir in dirs)
                {
                    string newDestinationDir = Path.Combine(destinationDir, subDir.Name);
                    CopyDirectory(subDir.FullName, newDestinationDir, true);
                }
            }
            return true;
        } // CopyDirectory

        //// https://stackoverflow.com/questions/2633628/can-i-get-command-line-arguments-of-other-processes-from-net-c
        //private static string GetCommandLine(this Process process)
        //{
        // // ManagementObjectSearcher is no found in System.Management on NT8 either ...
        //    using (System.Management.ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT CommandLine FROM Win32_Process WHERE ProcessId = " + process.Id))
        //    using (ManagementObjectCollection objects = searcher.Get())
        //    {
        //        return objects.Cast<ManagementBaseObject>().SingleOrDefault()?["CommandLine"]?.ToString();
        //    }

        //}

        // https://stackoverflow.com/questions/138043/find-the-next-tcp-port-in-net
        static int FreeTcpPort()
        {
            TcpListener l = new TcpListener(IPAddress.Loopback, 0);
            l.Start();
            int port = ((IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();
            return port;
        }

        static void Main(string[] args)
        {

            // https://stackoverflow.com/questions/2633628/can-i-get-command-line-arguments-of-other-processes-from-net-c
            if (false)
            {
                Console.WriteLine(" Process.GetProcessesByName('python.exe'):");
                // Process[] processList = Process.GetProcesses();
                foreach (var process in Process.GetProcessesByName("python"))
                {
                    try
                    {
                        Console.WriteLine("process.StartInfo.Arguments:" + process.StartInfo.Arguments); // always an empty string.
                        //Console.WriteLine("process.StartInfo.Arguments:" + GetCommandLine(process); // always an empty string.
                    }
                    catch (Win32Exception ex) when ((uint)ex.ErrorCode == 0x80004005)
                    {
                        // Intentionally empty - no security access to the process.
                    }
                    catch (InvalidOperationException)
                    {
                        // Intentionally empty - the process exited before getting details.
                    }

                } // foreach
            }


            if (false) // drive letter  as argument
            {
                if (args.Length == 0)
                {
                    Console.WriteLine("Please provide the RAM drive letter in the command line. Press CR ... ");
                    Console.ReadKey();
                    return;
                }
                drive = Convert.ToChar(args[0].ToUpper());
            }

            Console.Write("RAM drive letter: ");
            ConsoleKeyInfo cky = Console.ReadKey(true/*do not display*/); // Console.ReadKey(false); /
            drive = Convert.ToChar(cky.Key.ToString().ToUpper());
            Console.WriteLine(drive);

            //// Start with minimized window
            //IntPtr handle = Process.GetCurrentProcess().MainWindowHandle;
            //ShowWindow(handle, 6);

           // drive = Char.ToUpper(drive);
            string targetFilePath = drive + @":\python.exe";

            // If drive != 'c' means we have a RAM drive; we copy all Python stuff to the root of 'drive', if not already there
            // ----------------------------------------------------------------------------------------------------
            // check if python.exe is already in the RAM drive
            string pythonPath = ""; // GetPythonPath();
            if (File.Exists(targetFilePath) == false) //   Z:\python.exe does not exist
            {
                pythonPath = GetPythonPath(); // C:\Users\...AppData\Local\Programs\Python\Python312\python.exe

                if (drive != 'C') // we have a RAM drive; we copy Python.exe to the root of 'drive'
                {
                    FileInfo file = new FileInfo(pythonPath); // C:\Users\...AppData\Local\Programs\Python\Python312\python.exe

                    //string targetFilePath = drive +@":\python.exe";
                    file.CopyTo(targetFilePath);
                    Console.WriteLine("\t" + targetFilePath + " copied..." );
                    pythonPath = targetFilePath; //Z:\python.exe 
                }
            }
            else
            { // Z:\python.exe exists
                Console.WriteLine("\t" + targetFilePath + " already exists...");
                pythonPath = targetFilePath;  // Z:\python.exe 
            }

            // --------------   Entry rules  --------------------
            // copy the .py file and "several.13barJSONsamples.txt"  to the RAM drive
            string scriptPath = modelsFolder + "\\" + scriptName; // 'C:\Users\...\Documents\EntryModels_v3\EntryDirectionPredictionTCPv3.py'
            string dummyJSONfilePath = modelsFolder + "\\" + dummyJSONname; // 'C:\Users\...\Documents\EntryModels_v3\"several.13barJSONsamples.txt"'
            if (drive != 'C')
            {
                if (File.Exists(scriptPath) == false) // in  'C:\Users\...\Documents\EntryModels_v3\EntryDirectionPredictionTCPv3.py'
                {
                    Console.WriteLine("File " + scriptPath + " does not exist" +
                    "\n" +
                    "Please manually copy it to that location, then reload. Press CR ...");
                    Console.ReadKey();
                    return;
                }

                if (File.Exists(dummyJSONfilePath) == false) // in  'C:\Users\...\Documents\EntryModels_v3\"several.13barJSONsamples.txt"'
                {
                    Console.Write("File " + dummyJSONfilePath + " does not exist" +
                    "\n" +
                    "Please manually copy it to that location, then reload. Press CR ...");
                    Console.ReadKey();
                    return;
                } else // copy the file to the root of RAM drive 
                {
                    FileInfo file = new FileInfo(dummyJSONfilePath);
                    file.CopyTo(drive + @":\"+ dummyJSONname, true/*overwrite*/);
                    Console.WriteLine("\t" + dummyJSONname + " copied...");
                    dummyJSONfilePath = drive + @":\" + dummyJSONname; //Z:\"several.13barJSONsamples.txt"
                }

                string sourceDir = modelsFolder;
                string destinationDir = drive + ":\\" + modelsFolderName; // the root of 'drive'

                // Check if the destinationDir  exists
                var dir = new DirectoryInfo(destinationDir);
                if (!dir.Exists) // 
                {
                    //if (CopyDirectory(sourceDir, destinationDir, false /* recursive*/)) // we do not have sub-folders in EntryModels_v3
                    if (CopyDirectory(sourceDir, destinationDir, true /* recursive*/)) // we do have sub-folders in EntryModels_v3
                    {
                        modelsFolder = destinationDir;  // drive\EntryModels_v3
                        Console.WriteLine("\t" + modelsFolder + " folder  copied...");
                    }
                    else
                    {
                        Console.WriteLine("Attempt to copy folder " + modelsFolder + " to " + destinationDir + "\n" + "failed." +
                        "\n" +
                        "Please manually perform this operation, then reload NT8 strategy/indicator",/*this.Name+*/" EntryPrediction models folder transfer error");
                        Console.ReadKey();
                        return;
                    }

                }//  copy EntryModels_v3 folder
                else
                {
                    Console.WriteLine("\t" + dir.FullName + " folder already exists...");
                    modelsFolder = dir.FullName;
                }

                scriptPath = modelsFolder + "\\" + scriptName; // =  "\\EntryDirectionPredictionPipes.py";

                if (File.Exists(scriptPath) == false)
                {
                    FileInfo file = new FileInfo(scriptPath);
                    file.CopyTo(drive + @":\" + scriptPath /* == @":\EntryDirectionPredictionTCPv3.py"*/);
                    Console.WriteLine("\t" + drive + scriptPath + " copied...");

                }
                else
                {
                    Console.WriteLine("\t" + scriptPath + " already exists...");
                }

            }
            else // drive.ToUpper() == 'C'
            {
                // file EntryDirectionPredictionTCPv3.py needs to be on C:\ drive, in EntryModels_v3 folder 
                if (File.Exists(scriptPath) == false)
                {
                    Console.WriteLine("File "+ scriptName+" needs to be in " + modelsFolder + "folder." +
                    "\n" +
                    "Please manually copy it to that location, then reload NT8 strategy/indicator",/*this.Name+*/" Python script location error");
                    Console.ReadKey();
                    return;
                }

                pythonPath = GetPythonPath(); // of python.exe
            }

			//------------------------------------------------------
            // --------------   Exit rules  --------------------
            // copy the .py file and dummyExitData.json  to the RAM drive
            string scriptPathExit = modelsFolderExit + "\\" + scriptNameExit; // 'C:\Users\...\Documents\fitted_models\TickDirectionPrediction.py'
            string dummyJSONfilePathExit = modelsFolderExit + "\\" + dummyJSONnameExit; // 'C:\Users\...\Documents\fitted_models\dummyExitData.json'
            if (drive != 'C')
            {
                if (File.Exists(scriptPathExit) == false) // in  'C:\Users\...\Documents\fitted_models\TickDirectionPrediction.py'
                {
                    Console.WriteLine("File " + scriptPathExit + " does not exist" +
                    "\n" +
                    "Please manually copy it to that location, then reload. Press CR ...");
                    Console.ReadKey();
                    return;
                }

                if (File.Exists(dummyJSONfilePathExit) == false) // in  'C:\Users\...\Documents\fitted_models\dummyExitData.json'
                {
                    Console.Write("File " + dummyJSONfilePathExit + " does not exist" +
                    "\n" +
                    "Please manually copy it to that location, then reload. Press CR ...");
                    Console.ReadKey();
                    return;
                }
                else // copy two JSON files to the root of RAM drive 
                {
                    FileInfo file = new FileInfo(dummyJSONfilePathExit);
                    file.CopyTo(drive + @":\" + dummyJSONnameExit, true/*overwrite*/);
                    Console.WriteLine("\t" + dummyJSONnameExit + " copied to RAM drive root...");
                    dummyJSONfilePathExit = drive + @":\" + dummyJSONnameExit; //Z:\"several.13barJSONsamples.txt"
                }

                if (File.Exists(columnsPath) == false) // in  'C:\Users\...\Documents\fitted_models\columns.json'
                {
                    Console.Write("File " + columnsPath + " does not exist" +
                    "\n" +
                    "Please manually copy it to that location, then reload. Press CR ...");
                    Console.ReadKey();
                    return;
                }
                else // copy two JSON files to the root of RAM drive 
                {
                    FileInfo file = new FileInfo(columnsPath);
                    file.CopyTo(drive + @":\" + "columns.json", true/*overwrite*/);
                    Console.WriteLine("\t" +"columns.json" + " copied to RAM drive root...");
                    columnsPath = drive + @":\" + "columns.json"; //Z:\columns.json
                }

                string sourceDirExit = modelsFolderExit;
                string destinationDirExit = drive + ":\\" + modelsFolderNameExit; // the root of 'drive'

                // Check if the destinationDir  exists
                var dir = new DirectoryInfo(destinationDirExit);
                if (!dir.Exists) // 
                {
                    if (CopyDirectory(sourceDirExit, destinationDirExit, true /* recursive*/)) // we DO have sub-folders in fitted_models
                    {
                        modelsFolderExit = destinationDirExit;  // drive\fitted_models
                        Console.WriteLine("\t" + modelsFolderExit + " folder  copied...");
                    }
                    else
                    {
                        Console.WriteLine("Attempt to copy folder " + modelsFolderExit + " to " + destinationDirExit + "\n" + "failed." +
                        "\n" +
                        "Please manually perform this operation, then reload NT8 strategy/indicator",/*this.Name+*/" ExitPrediction models folder transfer error");
                        Console.ReadKey();
                        return;
                    }

                    
                }//  copy fittedt_models folder
                else
                {
                    Console.WriteLine("\t" + dir.FullName + " folder already exists...");
                    modelsFolderExit = dir.FullName;
                  
                }

               
                scriptPathExit = modelsFolderExit + "\\" + scriptNameExit; // =  "\\TickDirectionPrediction.py";

                if (File.Exists(scriptPathExit) == false)
                {
                    FileInfo file = new FileInfo(scriptPathExit);
                    file.CopyTo(drive + @":\" + scriptPathExit /* == @":\TickDirectionPrediction.py"*/);
                    Console.WriteLine("\t" + drive + scriptPathExit + " copied...");

                }
                else
                {
                    Console.WriteLine("\t" + scriptPathExit + " already exists...");
                }

            }
            else // drive.ToUpper() == 'C'
            {
                // file TickDirectionPrediction.py needs to be on C:\ drive, in fitted_models folder 
                if (File.Exists(scriptPathExit) == false)
                {
                    Console.WriteLine("File "+ scriptNameExit+" needs to be in " + modelsFolderExit + "folder." +
                    "\n" +
                    "Please manually copy it to that location, then reload NT8 strategy/indicator",/*this.Name+*/" Python script location error");
                    Console.ReadKey();
                    return;
                }

                pythonPath = GetPythonPath(); // of python.exe
            }


            // ------------  Entry Rules Server  -------------------
            residentPython = new System.Diagnostics.Process(); // <-----   Process object
                                                               //		residentPython.StartInfo.FileName = @"C:\Users\fawcett10\AppData\Local\Programs\Python\Python312\python.exe";  // <-----  Executable name ( Windows ) or binary (Linux/MacOS)
            ProcessStartInfo residentPython_StartInfo = new ProcessStartInfo();
            residentPython_StartInfo.FileName = pythonPath;  // <-----  Executable name ( Windows ) or binary (Linux/MacOS)
                                                             //		residentPython.StartInfo.Arguments =@"G:\rentacoder\Bill_upwork\Pipes\Method2\main.py";  // <----- Python file to be executed by the Python executable
            if (false) PORT = FreeTcpPort();
            residentPython_StartInfo.Arguments = scriptPath  // <----- .py  file to be executed by the Python executable
                                        + " " + HOST // 1st .py argument: IP
                                        + " " + PORT;  // 2nd .py argument: port #

            // redirect python.exe print() to Nt8 Output window							
            residentPython_StartInfo.RedirectStandardError = false; //true;
            residentPython_StartInfo.RedirectStandardOutput = false; //true;
            residentPython_StartInfo.UseShellExecute = false;
            residentPython_StartInfo.CreateNoWindow = false; // true;
            residentPython_StartInfo.WindowStyle = ProcessWindowStyle.Minimized; // no effect when CreateNoWindow = false
            residentPython_StartInfo.WindowStyle = ProcessWindowStyle.Hidden; // no effect when CreateNoWindow = false
            residentPython.EnableRaisingEvents = true;

            Console.WriteLine("\n" + "Starting Python.exe ..."); // + " CurrentThread.Name='" + System.Threading.Thread.CurrentThread.Name + "' ID:" + System.Threading.Thread.CurrentThread.ManagedThreadId);
            Console.WriteLine("\t path '" + pythonPath + "'");
            Console.WriteLine("\t script & arguments: " + residentPython_StartInfo.Arguments);

            Console.WriteLine("\n" + "Command line:");
            Console.WriteLine( pythonPath + " " + residentPython_StartInfo.Arguments);

            // Keep python.exe in RAM as long as this console window stays opened
            // https://stackoverflow.com/questions/6946008/keep-child-process-alive-after-parent-exits
            //  Process childProcess = Process.Start(residentPython.StartInfo); 
            residentPython = Process.Start(residentPython_StartInfo);

            // if (false) Console.WriteLine("\n" + "Python server Process.Id: " + childProcess.Id);

            //Console.WriteLine("\n" + "Python server MainModule.ModuleName: " + childProcess.MainModule.ModuleName);
            Console.WriteLine("\n" + "Wait for entry Python server to load in RAM ...");
            //Console.ReadKey();

            //return;

            DateTime dt0 = DateTime.Now;
            try
            {
                if (false) residentPython.Start();  // <---- Start python.exe 


                // wait max. 5sec for python.exe  and .py to be loaded in RAM, along with their import dependencies 
                bool pythonLoaded = false;
                while (true)
                {
                    string s = "";
                    try
                    {
                        s = residentPython.MainModule.FileName; // = python.exe
                        pythonLoaded = true;
                        break;
                    }
                    catch (Exception e) { }
                    if ((DateTime.Now - dt0).TotalMilliseconds > 5000) break;
                } // while
                if (pythonLoaded == false)
                {
                    Console.Write(" Resident python.exe  " +
                     residentPython.Id + " could not be loaded in " + (DateTime.Now - dt0).TotalMilliseconds + "ms" +
                     "\n" + "Please close then reload this application...");
                    Console.ReadKey();
                    return;
                }
            }
            catch (Exception e1)
            {
                Console.Write("python.exe caught exeption " + e1);
                Console.ReadKey();
                return;
            } // try & catch

            Console.WriteLine("Resident python.exe, process ID " + residentPython.Id +
                " loaded in " + (DateTime.Now - dt0).TotalMilliseconds + "ms"); // +
                                                                                //", started at " + residentPython.StartTime.ToString("o").Substring(11/*IndexOf("T")*/));

            // -------------------------------testing port on entry server connection C# <---> Python
            //                               ------------------------------------
            // specific port
            int portNo = PORT; // Int32.Parse(PORT);

            dt0 = DateTime.Now;
            int waitTimeMs = 10 * 1000; // 10 seconds
            bool connectionOK = false;
            while (true)
            {
                try
                {
                    Console.WriteLine("\tchecking port " + portNo + "...  ");

                    //// Example in Python : https://github.com/Allenci/Python-TCP-UDP-Listener/blob/master/tcpListener.py
                    //// Listens for connections from TCP network clients.
                    ////  https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets.tcplistener?redirectedfrom=MSDN&view=netframework-4.7.2
                    //TcpListener server = null;
                    //server = new TcpListener(IPAddress.Parse(HOST), portNo);

                    using (TcpClient client = new TcpClient(HOST, portNo)) // SocketException (0x80004005): No connection could be made because the target machine actively refused it 127.0.0.1:20010
                    {
                        string message = "ES" + " " + modelsFolder + " " + dummyJSONfilePath; //  "Z:\\AIdata.json";
                        // Translate the passed message into ASCII and store it as a Byte array.
                        Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);

                        // Get a client stream for reading and writing.
                        NetworkStream stream = client.GetStream();

                        // Send the message to the connected TcpServer.
                        stream.Write(data, 0, data.Length);

                        Console.WriteLine("Sent: {0}", message);

                        // Receive the server response.

                        // Buffer to store the response bytes.
                        data = new Byte[256];

                        // String to store the response ASCII representation.
                        String responseData = String.Empty;

                        // Read the first batch of the TcpServer response bytes.
                        Int32 bytes = stream.Read(data, 0, data.Length);
                        responseData = System.Text.Encoding.ASCII.GetString(data, 0, bytes);
                        Console.WriteLine("Received: {0}", responseData);
						// eg: Received: {'action': 2, 'confidence': 0.46586324347976327, 'position_size': 0, 'stop_loss': 0, 'timestamp': Timestamp('2025-06-23 20:11:16.391923')}
						// 0-Enter Long, 1-EnterShort, 2-NoTrade
						// By further selection (see conn.sendall(str(trade_signal.get("action")).encode()) in C:\Users\Administrator\Documents\EntryModels_v3\EntryDirectionPredictionTCPv3.py
						// responseData = 0,1,or 2 as a string

                        // Explicit close is not necessary since TcpClient.Dispose() will be
                        // called automatically.
                        // stream.Close();
                        // client.Close();
                        connectionOK = true;
                        break;
                    }
                }
                catch (Exception e) { }
                if ((DateTime.Now - dt0).TotalMilliseconds > waitTimeMs) break;
            } // while

            if (connectionOK == false)
            {
                Console.Write(" Entry rules Python server, python.exe ID  " +
                 residentPython.Id + ", did not respond in " + (DateTime.Now - dt0).TotalMilliseconds + "ms" +
                 "\n" + "Please close then reload this application...");
                Console.ReadKey();
                return;
            }
            else
            {
                Console.WriteLine(" Entry rules Python server, python.exe ID  " +
                 residentPython.Id + ", responded after  " + (DateTime.Now - dt0).TotalMilliseconds + "ms \n");
                //Console.WriteLine("\n" + "320 Press CR...");

                // Console.ReadKey();
            }

//goto finish;
            // ------------  Exit Rules Server  -------------------
            residentPythonExit = new System.Diagnostics.Process(); // <-----   Process object
            ProcessStartInfo exitServer_StartInfo = new ProcessStartInfo();
            exitServer_StartInfo.FileName = pythonPath;  // <-----  Executable name ( Windows ) or binary (Linux/MacOS)
                                                         //		residentPython.StartInfo.Arguments =@"G:\rentacoder\Bill_upwork\Pipes\Method2\main.py";  // <----- Python file to be executed by the Python executable
            if(false) PORTexit = FreeTcpPort();
            exitServer_StartInfo.Arguments = scriptPathExit  // <----- .py  file to be executed by the Python executable
                                        + " " + HOSTexit // 1st .py argument: IP
                                        + " " + PORTexit;  // 2nd .py argument: port #

            // redirect python.exe print() to Nt8 Output window							
            exitServer_StartInfo.RedirectStandardError = false; //true;
            exitServer_StartInfo.RedirectStandardOutput = false; //true;
            exitServer_StartInfo.UseShellExecute = false;
            exitServer_StartInfo.CreateNoWindow = false; // true;
            exitServer_StartInfo.WindowStyle = ProcessWindowStyle.Minimized; // no effect when CreateNoWindow = false
            exitServer_StartInfo.WindowStyle = ProcessWindowStyle.Hidden; // no effect when CreateNoWindow = false
            residentPythonExit.EnableRaisingEvents = true;

            Console.WriteLine("\n" + "Starting Python.exe ..."); // + " CurrentThread.Name='" + System.Threading.Thread.CurrentThread.Name + "' ID:" + System.Threading.Thread.CurrentThread.ManagedThreadId);
            Console.WriteLine("\t path '" + pythonPath + "'");
            Console.WriteLine("\t script & arguments: " + exitServer_StartInfo.Arguments);

            Console.WriteLine("\n" + "Command line:");
            Console.WriteLine(pythonPath + " " + exitServer_StartInfo.Arguments);

            // Keep python.exe in RAM as long as this console window stays opened
            // https://stackoverflow.com/questions/6946008/keep-child-process-alive-after-parent-exits
            //  Process childProcess = Process.Start(residentPython.StartInfo); 
            residentPythonExit = Process.Start(exitServer_StartInfo);

            Console.WriteLine("\n" + "Wait for exit Python server to load in RAM ...");

           /* DateTime*/ dt0 = DateTime.Now;
            try
            {
                //if (false) residentPython.Start();  // <---- Start python.exe 

                // wait max. 5sec for python.exe  and .py to be loaded in RAM, along with their import dependencies 
                bool pythonLoaded = false;
                while (true)
                {
                    string s = "";
                    try
                    {
                        s = residentPythonExit.MainModule.FileName; // = python.exe
                        pythonLoaded = true;
                        break;
                    }
                    catch (Exception e) { }
                    if ((DateTime.Now - dt0).TotalMilliseconds > 5000) break;
                } // while
                if (pythonLoaded == false)
                {
                    Console.Write(" Resident python.exe  " +
                     residentPython.Id + " could not be loaded in " + (DateTime.Now - dt0).TotalMilliseconds + "ms" +
                     "\n" + "Please close then reload this application...");
                    Console.ReadKey();
                    return;
                }
            }
            catch (Exception e1)
            {
                Console.Write("python.exe caught exeption " + e1);
                Console.ReadKey();
                return;
            } // try & catch

            Console.WriteLine("Resident python.exe, process ID " + residentPython.Id +
                " loaded in " + (DateTime.Now - dt0).TotalMilliseconds + "ms"); // +
                                                                                //", started at " + residentPython.StartTime.ToString("o").Substring(11/*IndexOf("T")*/));
 goto testExitport; // skip the ping

            // -------------------------------testing entry server connection C# <---> Python
            //                               ------------------------------------
            Console.WriteLine("\nTesting the entry rules server connection C# <---> Python server " + HOST + ":" + PORT + " ..."); 

            // if .py fails then python.exe does not exist anymore
            Process[] processList = Process.GetProcesses();
            if (processList.FirstOrDefault(pr => pr.Id == residentPython.Id) == null)
            {
                Console.WriteLine("\n resident python.exe quit meantime, possible due to an error in .py script  " + DateTime.Now.ToString("o").Substring(11/*IndexOf("T")*/));
                Console.WriteLine("\n ith the cmd command from Output one may detect the error  in .py script  ");
                Console.ReadKey();
                return;
            }

            /*bool*/ connectionOK = false;

            // ping the server
            Console.Write("\tPing the entry rules Python server ... ");
            System.Net.NetworkInformation.Ping ping = new System.Net.NetworkInformation.Ping();
            IPAddress ipAdress = IPAddress.Parse(HOST); //  new IPAddress(new byte[] { 127, 0, 0, 1 });
            System.Net.NetworkInformation.PingReply reply = ping.Send(ipAdress, 3000/*ms*/); // An invalid IP address was specified.
                                                                                         		
            if (reply == null)
            {
                Console.WriteLine("  failed, reply=null \nClose and try again");
                Console.ReadKey();
                return;
            }

            if (reply.Status == System.Net.NetworkInformation.IPStatus.Success)
            {
                Console.WriteLine(  reply.Status);
                connectionOK = true;
            }
            else
            {
                Console.WriteLine("  failed  : " + reply.Status+"\n Close and try again");
                Console.ReadKey();
                return;
            }

            // -------------------------------testing exit rules  server connection C# <---> Python
            //                               ------------------------------------
            Console.WriteLine("\nTesting the exit rules server connection C# <---> Python server " + HOSTexit + ":" + PORTexit + " ...");

            // if .py fails then python.exe does not exist anymore
            //Process[] processList = Process.GetProcesses();
            if (processList.FirstOrDefault(pr => pr.Id == residentPythonExit.Id) == null)
            {
                Console.WriteLine("\n resident python.exe quit meantime, possible due to an error in .py script  " + DateTime.Now.ToString("o").Substring(11/*IndexOf("T")*/));
                Console.WriteLine("\n ith the cmd command from Output one may detect the error  in .py script  ");
                Console.ReadKey();
                return;
            }

            /*bool*/ connectionOK = false;

            // ping the server
            Console.Write("\tPing the exit rules Python server ... ");
            //System.Net.NetworkInformation.Ping ping = new System.Net.NetworkInformation.Ping();
            /*IPAddress*/ ipAdress = IPAddress.Parse(HOSTexit); //  new IPAddress(new byte[] { 127, 0, 0, 1 });
            /*System.Net.NetworkInformation.PingReply*/ reply = ping.Send(ipAdress, 3000/*ms*/); // An invalid IP address was specified.

            if (reply == null)
            {
                Console.WriteLine("  failed, reply=null \nClose and try again");
                Console.ReadKey();
                return;
            }

            if (reply.Status == System.Net.NetworkInformation.IPStatus.Success)
            {
                Console.WriteLine(reply.Status);
                connectionOK = true;
            }
            else
            {
                Console.WriteLine("  failed  : " + reply.Status + "\n Close and try again");
                Console.ReadKey();
                return;
            }

            //Console.ReadKey();

            //return;

testExitport:

            // -------------------------------testing port on exit server connection C# <---> Python: may not be successful outside NT8
            //                               ------------------------------------

            //return;


            // specific port
            /*int*/ portNo = PORTexit; // Int32.Parse(PORT);

            dt0 = DateTime.Now;
            /*int*/ waitTimeMs = 10 * 1000; // 10 seconds
            connectionOK = false;
            while (true)
            {
                try
                {
                    Console.WriteLine("\tchecking exit rules port " + portNo + "...  ");

                    //// Example in Python : https://github.com/Allenci/Python-TCP-UDP-Listener/blob/master/tcpListener.py
                    //// Listens for connections from TCP network clients.
                    ////  https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets.tcplistener?redirectedfrom=MSDN&view=netframework-4.7.2
                    //TcpListener server = null;
                    //server = new TcpListener(IPAddress.Parse(HOST), portNo);

                    using (TcpClient client = new TcpClient(HOSTexit, portNo)) // SocketException (0x80004005): No connection could be made because the target machine actively refused it 127.0.0.1:20010
                    {
                        string message = dummyJSONfilePathExit + " " + columnsPath+" " + "Short PV1 2 "+ drive + ":\\" + modelsFolderNameExit + "\\ym";

                        // Translate the passed message into ASCII and store it as a Byte array.
                        Byte[] data = System.Text.Encoding.ASCII.GetBytes(message);

                        // Get a client stream for reading and writing.
                        NetworkStream stream = client.GetStream();

                        // Send the message to the connected TcpServer.
                        stream.Write(data, 0, data.Length);

                        Console.WriteLine("Sent: {0}", message);

                        // Receive the server response.

                        // Buffer to store the response bytes.
                        data = new Byte[256];

                        // String to store the response ASCII representation.
                        String responseData = String.Empty;

                        // Read the first batch of the TcpServer response bytes.
                        Int32 bytes = stream.Read(data, 0, data.Length);
                        responseData = System.Text.Encoding.ASCII.GetString(data, 0, bytes);
                        Console.WriteLine("Received: {0}", responseData);

                        // Explicit close is not necessary since TcpClient.Dispose() will be
                        // called automatically.
                        // stream.Close();
                        // client.Close();
                        connectionOK = true;
                        break;
                    }
                }
                catch (Exception e) { }
                if ((DateTime.Now - dt0).TotalMilliseconds > waitTimeMs) break;
            } // while

            if (connectionOK == false)
            {
                Console.Write(" Exit rules Python server, python.exe ID  " +
                 residentPython.Id + ", did not respond in " + (DateTime.Now - dt0).TotalMilliseconds + "ms" +
                 "\n" + "Please close then reload this application...");
                Console.ReadKey();
            }
            else
            {
                Console.WriteLine(" Exit rules Python server, python.exe ID  " +
                 residentPython.Id + ", responded after  " + (DateTime.Now - dt0).TotalMilliseconds + "ms \n");
                //Console.WriteLine("\n" + "320 Press CR...");

               // Console.ReadKey();
            }
finish:
            // Minimize the window if everything was OK
            if (connectionOK)
            {
                IntPtr handle = Process.GetCurrentProcess().MainWindowHandle;
                ShowWindow(handle, 6);
            }

        } // Main



    } // class Program
} // namespace EntryPythonServer
