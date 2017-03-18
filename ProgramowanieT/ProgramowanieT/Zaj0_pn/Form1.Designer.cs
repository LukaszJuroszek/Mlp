namespace Zaj0_pn
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btnIloraz = new System.Windows.Forms.Button();
            this.btnIloczyn = new System.Windows.Forms.Button();
            this.btMinus = new System.Windows.Forms.Button();
            this.btnPlus = new System.Windows.Forms.Button();
            this.txtLiczbaB = new System.Windows.Forms.TextBox();
            this.txtLiczbaA = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnIloraz);
            this.groupBox1.Controls.Add(this.btnIloczyn);
            this.groupBox1.Controls.Add(this.btMinus);
            this.groupBox1.Controls.Add(this.btnPlus);
            this.groupBox1.Controls.Add(this.txtLiczbaB);
            this.groupBox1.Controls.Add(this.txtLiczbaA);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Location = new System.Drawing.Point(13, 13);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(247, 84);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Kalkulator";
            // 
            // btnIloraz
            // 
            this.btnIloraz.Location = new System.Drawing.Point(200, 47);
            this.btnIloraz.Name = "btnIloraz";
            this.btnIloraz.Size = new System.Drawing.Size(26, 23);
            this.btnIloraz.TabIndex = 7;
            this.btnIloraz.Text = "/";
            this.btnIloraz.UseVisualStyleBackColor = true;
            this.btnIloraz.Click += new System.EventHandler(this.Wybor);
            // 
            // btnIloczyn
            // 
            this.btnIloczyn.Location = new System.Drawing.Point(167, 47);
            this.btnIloczyn.Name = "btnIloczyn";
            this.btnIloczyn.Size = new System.Drawing.Size(27, 23);
            this.btnIloczyn.TabIndex = 6;
            this.btnIloczyn.Text = "*";
            this.btnIloczyn.UseVisualStyleBackColor = true;
            this.btnIloczyn.Click += new System.EventHandler(this.Wybor);
            // 
            // btMinus
            // 
            this.btMinus.Location = new System.Drawing.Point(200, 17);
            this.btMinus.Name = "btMinus";
            this.btMinus.Size = new System.Drawing.Size(26, 23);
            this.btMinus.TabIndex = 5;
            this.btMinus.Text = "-";
            this.btMinus.UseVisualStyleBackColor = true;
            this.btMinus.Click += new System.EventHandler(this.Wybor);
            // 
            // btnPlus
            // 
            this.btnPlus.Location = new System.Drawing.Point(167, 18);
            this.btnPlus.Name = "btnPlus";
            this.btnPlus.Size = new System.Drawing.Size(27, 23);
            this.btnPlus.TabIndex = 4;
            this.btnPlus.Text = "+";
            this.btnPlus.UseVisualStyleBackColor = true;
            this.btnPlus.Click += new System.EventHandler(this.Wybor);
            // 
            // txtLiczbaB
            // 
            this.txtLiczbaB.Location = new System.Drawing.Point(61, 49);
            this.txtLiczbaB.Name = "txtLiczbaB";
            this.txtLiczbaB.Size = new System.Drawing.Size(100, 20);
            this.txtLiczbaB.TabIndex = 3;
            // 
            // txtLiczbaA
            // 
            this.txtLiczbaA.Location = new System.Drawing.Point(61, 20);
            this.txtLiczbaA.Name = "txtLiczbaA";
            this.txtLiczbaA.Size = new System.Drawing.Size(100, 20);
            this.txtLiczbaA.TabIndex = 2;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(7, 52);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(47, 13);
            this.label2.TabIndex = 1;
            this.label2.Text = "Liczba b";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(6, 23);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(47, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Liczba a";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(269, 101);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtLiczbaB;
        private System.Windows.Forms.TextBox txtLiczbaA;
        private System.Windows.Forms.Button btnIloraz;
        private System.Windows.Forms.Button btnIloczyn;
        private System.Windows.Forms.Button btMinus;
        private System.Windows.Forms.Button btnPlus;
    }
}

