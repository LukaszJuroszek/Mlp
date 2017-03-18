using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Zaj1_pn
{
    static class Program
    {
        //public static string ConnStr
        //{
        //    get
        //    {
        //        return "Server = DESKTOP-67V6J07; Database = Zaj1; Trusted_Connection = True; ";
        //    }
        //}

        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
