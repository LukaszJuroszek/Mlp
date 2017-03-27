using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace T4.Model
{
    public class Project
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public virtual ICollection<TeamMeber> TeamMeber { get; set; }
        public Project()
        {
            TeamMeber = new HashSet<TeamMeber>();
        }
    }
}
