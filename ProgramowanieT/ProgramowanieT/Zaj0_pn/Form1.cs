using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Zaj0_pn
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Wybor(object sender, EventArgs e)
        {
            try
            {
                double a = double.Parse(txtLiczbaA.Text);
                double b = double.Parse(txtLiczbaB.Text);

                string operation = ((sender as Button).Text).ToString();

                Dictionary<string, Func<double, double, double>> operations = new Dictionary<string, Func<double, double, double>>()
                {
                    { "+",(n, m) => n + m },
                    { "-",(n, m) => n - m },
                    { "*",(n, m) => n * m },
                    { "/",(n, m) => n / m }
                };

                MessageBox.Show(operations[operation](a, b).ToString(), "Wynik", MessageBoxButtons.OK,MessageBoxIcon.Information);
            }
            catch (Exception)
            {
                MessageBox.Show("Nieprawidlowe dane","Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        

        }
    }
}
