using EventsDal.Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Configuration;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Zaj1_pn
{
    public partial class FrmAddEdit : Form
    {
        public FrmAddEdit()
        {
            InitializeComponent();
        }

        public FrmAddEdit(string title,string description,DateTime when) : this()
        {
            textBoxTitle.Text = title;
            textBoxOpis.Text = description;
            dtpWhen.Value = when;
        }
        public Event GetEventToSave()
        {
            //var connStr = ConfigurationManager.ConnectionStrings["Events"].ConnectionString;
            //using (var conn = new SqlConnection(connStr))
            //{
            //    conn.Open();
            //    SqlCommand cmd = new SqlCommand("INSERT INTO Event ([Title], [Description], [When]) VALUES (@Title, @Description, @When)",conn);
            //    cmd.Parameters.AddWithValue("@Title",textBoxTitle.Text);
            //    cmd.Parameters.AddWithValue("@Description",textBoxOpis.Text);
            //    cmd.Parameters.AddWithValue("@When",dtpWhen.Value);
            //    cmd.ExecuteNonQuery();
            return new Event { Title = textBoxTitle.Text,Description = textBoxOpis.Text,When = dtpWhen.Value };
            //}
        }
        public Event GetEventToEdit(int id)
        {
            //var connStr = ConfigurationManager.ConnectionStrings["Events"].ConnectionString;
            //using (var conn = new SqlConnection(connStr))
            //{
            //    conn.Open();
            //    var cmd = new SqlCommand("UPDATE Event SET [Title]=(@Title), [Description]=(@Description), [When]=(@When) WHERE [Id]=(@Id)",conn);
            //    cmd.Parameters.AddWithValue("@Title",);
            //    cmd.Parameters.AddWithValue("@Description",textBoxOpis.Text);
            //    cmd.Parameters.AddWithValue("@When",dtpWhen.Value);
            //    cmd.Parameters.AddWithValue("@Id",id);
            //    cmd.ExecuteNonQuery();
            //}
            //    cmd.Parameters.AddWithValue("@Description",textBoxOpis.Text);
            //    cmd.Parameters.AddWithValue("@When",dtpWhen.Value);
            return new Event { Id = id,Title = textBoxTitle.Text,Description = textBoxOpis.Text,When = dtpWhen.Value };
        }
    }
}
