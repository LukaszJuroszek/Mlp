using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DapperTest
{
    public class Produkt
    {
        public int ID_Produkt { get; set; }
        public string Nazwa { get; set; }
        public string Jednostka_Produktu { get; set; }
        public override string ToString()
        {

            var sb = new StringBuilder();
            sb.Append($"{ID_Produkt}, {Nazwa}, {Jednostka_Produktu}");
            return sb.ToString();
        }
    }
}
