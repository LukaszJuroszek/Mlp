using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace T4.Model
{
    public enum MemberType
    {
        Developer,
        TeamLeader,
        ScrumMaster,
        ProduktOwner
    }
    public class TeamMeber
    {

        public int Id { get; set; }
        public string Name { get; set; }
        public MemberType MemberType { get; set; }
        public Team Team { get; set; }
        public virtual ICollection<Project> Projects { get; set; }
        public TeamMeber()
        {
            Projects = new HashSet<Project>();
        }
    }
}
