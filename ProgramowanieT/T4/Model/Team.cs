using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace T4.Model
{
    public class Team
    {
        public int Id { get; set; }
        public string Name { get; set; }
        //musi być tutaj wirual-- 
        public virtual ICollection<TeamMeber> TeamMembers { get; set; }
        public Team()
        {
            TeamMembers = new HashSet<TeamMeber>();
        }
    }
}
