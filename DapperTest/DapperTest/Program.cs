using Dapper;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DapperTest
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var dBConnection = new SqlConnection(ConfigurationManager.ConnectionStrings["SqlConnection"].ConnectionString))
            {
                dBConnection.Open();
                var query = "Select * from Produkt";
                var resultQuery = dBConnection.Query<Produkt>(query);
                foreach (var item in resultQuery)
                {
                    if (( item.Jednostka_Produktu != "sztuki" ))
                        Console.WriteLine(item);
                    Console.WriteLine(item);
                }
            }
        }
    }
}
