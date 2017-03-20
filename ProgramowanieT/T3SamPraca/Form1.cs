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
        private int pictureCount = 20;
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
            var dirInfo = new DirectoryInfo("C:\\EkspertPlus\\01_05\\VisualStudio\\Gra\\Obrazki\\Animowane");
            animowane = dirInfo.GetFiles("*.gif");
            dirInfo = new DirectoryInfo("C:\\EkspertPlus\\01_05\\VisualStudio\\Gra\\Obrazki\\Zabawne");
            zabawne = dirInfo.GetFiles("*.bmp");
        }

        private void btnStart_Click(object sender, EventArgs level)
        {
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
                .Range(1, range)
                .OrderBy(x => Guid.NewGuid())
                .Take(pictureCount)
                .OrderBy(x => x)
                .ToArray();
        }
    }
}
