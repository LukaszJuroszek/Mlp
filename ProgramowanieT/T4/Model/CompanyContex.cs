using System.Data.Entity;

namespace T4.Model
{
    public class CompanyContext : DbContext
    {
        public DbSet<Team> Team { get; set; }
        public DbSet<TeamMeber> TeamMembers { get; set; }
        public DbSet<Project> Projects{ get; set; }
        public CompanyContext()
        {

        }

    }
}
