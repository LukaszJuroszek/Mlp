namespace Zaj1_pn
{
    partial class FrmAddEdit
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
            this.textBoxTitle = new System.Windows.Forms.TextBox();
            this.textBoxOpis = new System.Windows.Forms.TextBox();
            this.dtpWhen = new System.Windows.Forms.DateTimePicker();
            this.lblTitle = new System.Windows.Forms.Label();
            this.lblDescription = new System.Windows.Forms.Label();
            this.btnCancelAdd = new System.Windows.Forms.Button();
            this.btnSaveAdd = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // textBoxTitle
            // 
            this.textBoxTitle.Location = new System.Drawing.Point(61, 35);
            this.textBoxTitle.Name = "textBoxTitle";
            this.textBoxTitle.Size = new System.Drawing.Size(200, 20);
            this.textBoxTitle.TabIndex = 0;
            // 
            // textBoxOpis
            // 
            this.textBoxOpis.Location = new System.Drawing.Point(61, 87);
            this.textBoxOpis.Multiline = true;
            this.textBoxOpis.Name = "textBoxOpis";
            this.textBoxOpis.Size = new System.Drawing.Size(200, 103);
            this.textBoxOpis.TabIndex = 0;
            // 
            // dtpWhen
            // 
            this.dtpWhen.Location = new System.Drawing.Point(61, 216);
            this.dtpWhen.MinDate = new System.DateTime(1800, 1, 1, 0, 0, 0, 0);
            this.dtpWhen.Name = "dtpWhen";
            this.dtpWhen.Size = new System.Drawing.Size(200, 20);
            this.dtpWhen.TabIndex = 1;
            // 
            // lblTitle
            // 
            this.lblTitle.AutoSize = true;
            this.lblTitle.Location = new System.Drawing.Point(16, 38);
            this.lblTitle.Name = "lblTitle";
            this.lblTitle.Size = new System.Drawing.Size(30, 13);
            this.lblTitle.TabIndex = 2;
            this.lblTitle.Text = "Tytul";
            // 
            // lblDescription
            // 
            this.lblDescription.AutoSize = true;
            this.lblDescription.Location = new System.Drawing.Point(18, 90);
            this.lblDescription.Name = "lblDescription";
            this.lblDescription.Size = new System.Drawing.Size(28, 13);
            this.lblDescription.TabIndex = 3;
            this.lblDescription.Text = "Opis";
            // 
            // btnCancelAdd
            // 
            this.btnCancelAdd.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.btnCancelAdd.Location = new System.Drawing.Point(12, 277);
            this.btnCancelAdd.Name = "btnCancelAdd";
            this.btnCancelAdd.Size = new System.Drawing.Size(75, 23);
            this.btnCancelAdd.TabIndex = 4;
            this.btnCancelAdd.Text = "Anuluj";
            this.btnCancelAdd.UseVisualStyleBackColor = true;
            // 
            // btnSaveAdd
            // 
            this.btnSaveAdd.DialogResult = System.Windows.Forms.DialogResult.OK;
            this.btnSaveAdd.Location = new System.Drawing.Point(239, 277);
            this.btnSaveAdd.Name = "btnSaveAdd";
            this.btnSaveAdd.Size = new System.Drawing.Size(75, 23);
            this.btnSaveAdd.TabIndex = 5;
            this.btnSaveAdd.Text = "Zapisz";
            this.btnSaveAdd.UseVisualStyleBackColor = true;
            // 
            // FrmAddEdit
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(326, 312);
            this.Controls.Add(this.btnSaveAdd);
            this.Controls.Add(this.btnCancelAdd);
            this.Controls.Add(this.lblDescription);
            this.Controls.Add(this.lblTitle);
            this.Controls.Add(this.dtpWhen);
            this.Controls.Add(this.textBoxOpis);
            this.Controls.Add(this.textBoxTitle);
            this.Name = "FrmAddEdit";
            this.Text = "FrmAddEdit";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox textBoxTitle;
        private System.Windows.Forms.TextBox textBoxOpis;
        private System.Windows.Forms.DateTimePicker dtpWhen;
        private System.Windows.Forms.Label lblTitle;
        private System.Windows.Forms.Label lblDescription;
        private System.Windows.Forms.Button btnCancelAdd;
        private System.Windows.Forms.Button btnSaveAdd;
    }
}