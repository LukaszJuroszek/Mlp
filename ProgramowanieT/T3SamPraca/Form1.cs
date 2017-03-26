using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace T3SamPraca
{
    public partial class Form1 : Form
    {
        private int level = 20;
        private int pictureCount = 10;
        private int pairMatched = 0;
        private int picNumber1 = -1;
        private int picNumber2 = -1;
        private FileInfo[] animowane;
        private FileInfo[] zabawne;


        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            //var dirInfo = new DirectoryInfo("C:\\EkspertPlus\\01_05\\VisualStudio\\Gra\\Obrazki\\Animowane");
            //animowane = dirInfo.GetFiles("*.gif");
            var dirInfo = new DirectoryInfo(@"C:\Users\Arito\Source\Repos\programowanie4\ProgramowanieT\T3SamPraca");
            zabawne = dirInfo.GetFiles("*.bmp");
        }

        private void btnStart_Click(object sender, EventArgs level)
        {
            for (int i = 0; i < this.level; i++)
            {
                PictureBox picBox = (PictureBox)flpKontener.Controls[i];
                picBox.Tag = picBox.Image = null;
                picBox.BackColor = Color.Transparent;
                picBox.Enabled = true;
                pairMatched = 0;
                picNumber1 = picNumber2 = -1;
            }
            {
                if (łatwyToolStripMenuItem.Checked)
                    this.level = 8;
                else if (średniToolStripMenuItem.Checked)
                    this.level = 14;
                else this.level = 20;
                int[] randomImageNumber = Draw(this.level / 2, pictureCount);
                int[] randomPicBoxNumber = Draw(this.level, this.level);
                var k = 0;
                for (var i = 0; i < this.level / 2; i++)
                {
                    ((PictureBox)flpKontener.Controls[randomPicBoxNumber[k]]).Tag = randomImageNumber[i];
                    ((PictureBox)flpKontener.Controls[randomPicBoxNumber[k]]).BackColor = Color.AliceBlue;
                    ((PictureBox)flpKontener.Controls[randomPicBoxNumber[k + 1]]).Tag = randomImageNumber[i];
                    ((PictureBox)flpKontener.Controls[randomPicBoxNumber[k + 1]]).BackColor = Color.AliceBlue;
                    k += 2;
                }
            }
        }
        private int[] Draw(int pictureCount, int range)
        {
            return Enumerable
                .Range(0, range)
                .OrderBy(x => Guid.NewGuid())
                .Take(pictureCount)
                //.OrderBy(x => x)
                .ToArray();
        }

        private void pictureBox20_MouseEnter(object sender, EventArgs e)
        {
            var pictureBox = (PictureBox)sender;
            if (pictureBox.Tag == null)
                pictureBox.BackColor = Color.DarkGray;
        }

        private void pictureBox20_MouseLeave(object sender, EventArgs e)
        {
            var pictureBox = (PictureBox)sender;
            if (pictureBox.Tag == null)
                pictureBox.BackColor = Color.AliceBlue;
        }

        private void pictureBox20_Click(object sender, EventArgs e)
        {
            var currentPicBox = (PictureBox)sender;
            var currentPicBoxNumber = flpKontener.Controls.GetChildIndex(currentPicBox);
            if (currentPicBox.Tag != null)
            {
                if (currentPicBoxNumber != picNumber1 && currentPicBoxNumber != picNumber2)
                {
                    var pictureNumber = (int)currentPicBox.Tag;
                    if (animowaneToolStripMenuItem.Checked)
                        currentPicBox.Image =
                        Image.FromFile(animowane[pictureNumber].FullName);
                    else
                        currentPicBox.Image =
                        Image.FromFile(zabawne[pictureNumber].FullName);
                    if (picNumber1 == -1)
                        picNumber1 = currentPicBoxNumber;
                    else
                    if (picNumber2 == -1)
                    {
                        picNumber2 = currentPicBoxNumber;
                        if (pairMatched == level - 2)
                        {
                            DialogResult result = MessageBox.Show
                            (this, "Koniec gry.\nJeszcze raz to samo ?", "info",
                            MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                            if (result == DialogResult.Yes)
                                btnStart_Click(sender, e);
                        }
                    }
                    else
                    {
                        var picBox1 = (PictureBox)flpKontener.Controls[picNumber1];
                        var picBox2 = (PictureBox)flpKontener.Controls[picNumber2];
                        if (!((int)picBox1.Tag == (int)picBox2.Tag))
                        {
                            picBox1.Image = null;
                            picBox2.Image = null;
                        }
                        else
                        {
                            picBox1.Enabled = false;
                            picBox2.Enabled = false;
                            pairMatched += 2;
                        }
                        picNumber1 = currentPicBoxNumber;
                        picNumber2 = -1;
                    }
                }
            }
        }

        private void łatwyToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (łatwyToolStripMenuItem.Checked) return;
            łatwyToolStripMenuItem.Checked = true;
            średniToolStripMenuItem.Checked = trudnyToolStripMenuItem.Checked = false;
        }

        private void średniToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (średniToolStripMenuItem.Checked) return;
            średniToolStripMenuItem.Checked = true;
            łatwyToolStripMenuItem.Checked = trudnyToolStripMenuItem.Checked = false;
        }

        private void trudnyToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (trudnyToolStripMenuItem.Checked) return;
            trudnyToolStripMenuItem.Checked = true;
            łatwyToolStripMenuItem.Checked = średniToolStripMenuItem.Checked = false;
        }

        private void zabawneToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (zabawneToolStripMenuItem.Checked) return;
            zabawneToolStripMenuItem.Checked = true;
            animowaneToolStripMenuItem.Checked = false;
        }

        private void animowaneToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (animowaneToolStripMenuItem.Checked) return;
            animowaneToolStripMenuItem.Checked = true;
            zabawneToolStripMenuItem.Checked = false;
        }
    }
}
