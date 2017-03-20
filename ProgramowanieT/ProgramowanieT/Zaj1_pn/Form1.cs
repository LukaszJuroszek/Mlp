using EventsDal.Concrete;
using EventsDal.Model;
using System;
using System.Configuration;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Windows.Forms;
namespace Zaj1_pn
{
    public partial class Form1 : Form
    {
        private IEventRepository  _eventsRepository;
        public Form1()
        {
            InitializeComponent();
            //sql, xml
            //_eventsRepository = new SQLEventRepository();
            //_eventsRepository = new MemoryEventRepository();
            _eventsRepository = new XMLEventRepository();
        }
        public void Form1_Load(object sender,EventArgs e)
        {
            var events = _eventsRepository.GetAll();
            dgvEvents.DataSource = events.ToList();
            //string connStr = ConfigurationManager.ConnectionStrings["Events"].ConnectionString;
            //using (SqlConnection conn = new SqlConnection(connStr))
            //{
            //    conn.Open();
            //    SqlCommand cmd = new SqlCommand("SELECT * FROM Event", conn);
            //    var dr = cmd.ExecuteReader();
            //    DataTable dt = new DataTable();
            //    dt.Load(dr);
            //    dgvEvents.DataSource = dt;
            //}
        }
        private void btnAdd_Click(object sender,EventArgs e)
        {
            var frmAddEdit = new FrmAddEdit();
            if (frmAddEdit.ShowDialog() == DialogResult.OK)
            {
                _eventsRepository.Add(frmAddEdit.GetEventToSave());
                Form1_Load(sender,e);
            }
        }
        private void btnDelete_Click(object sender,EventArgs e)
        {
            foreach (var row in dgvEvents.SelectedRows.Cast<DataGridViewRow>())
            {
                var id = int.Parse(row.Cells[0].Value.ToString());
                _eventsRepository.Delete(id);
            }
            Form1_Load(sender,e);
        }
        private void btnEdit_Click(object sender,EventArgs e)
        {
            var id = int.Parse(dgvEvents.Rows[dgvEvents.SelectedCells[0].RowIndex].Cells[0].Value.ToString());
            var title = dgvEvents.Rows[dgvEvents.SelectedCells[0].RowIndex].Cells[1].Value.ToString();
            var desc = dgvEvents.Rows[dgvEvents.SelectedCells[0].RowIndex].Cells[2].Value.ToString();
            var when = DateTime.Parse(dgvEvents.Rows[dgvEvents.SelectedCells[0].RowIndex].Cells[3].Value.ToString());
            var frmAddEdit = new FrmAddEdit(title,desc,when);
            if (frmAddEdit.ShowDialog() == DialogResult.OK)
            {
                _eventsRepository.Edit(frmAddEdit.GetEventToEdit(id));
                Form1_Load(sender,e);
            }
        }
        private void Delete(int id)
        {
            var connStr = ConfigurationManager.ConnectionStrings["Events"].ConnectionString;
            using (var conn = new SqlConnection(connStr))
            {
                conn.Open();
                var cmd = new SqlCommand("DELETE FROM Event WHERE ([Id])=(@ID)",conn);
                cmd.Parameters.AddWithValue("@ID",id);
                cmd.ExecuteNonQuery();
            }
        }
        private void btnRefresh_Click(object sender,EventArgs e)
        {
            Form1_Load(sender,e);
        }
    }
}
