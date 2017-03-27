using System;
using T4.Model;

namespace T4
{
    class Program
    {
        //codefirst Entiti
        static void Main(string[] args)
        {
            string dataPath = AppDomain.CurrentDomain.BaseDirectory;
            AppDomain.CurrentDomain.SetData("DataDirectory", dataPath);
            using (var db = new CompanyContext())
            {
                db.Database.CreateIfNotExists();
            }
        }
    }
}
