using System.Data.Entity;

namespace T4.Model
{
    public class CompanyContext : DbContext
    {
        public DbSet<Team> Team { get; set; }
        public DbSet<TeamMeber> TeamMembers { get; set; }
        public DbSet<Project> Projects { get; set; }
        public CompanyContext(){}
        public static void SetNullTeamMembersOnTeamsDeleted(Database context)
        {
            ////https://social.msdn.microsoft.com/Forums/en-US/a3669916-5fbe-4e8b-8405-fa4c7ed04ffc/where-is-willnullondelete-or-willdefaultondelete?forum=adodotnetentityframework
            //http://stackoverflow.com/questions/25056031/does-entity-framework-support-when-deleted-a-record-and-set-the-foreign-key-to-n
            //Database.ExecuteSqlCommand("ALTER TABLE Teams DROP CONSTRAINT Teams_Constraint");
            context.ExecuteSqlCommand("ALTER TABLE dbo.TeamMebers ADD CONSTRAINT FK_Team_TeamId FOREIGN KEY(Team_Id) REFERENCES dbo.Teams(Id)  ON UPDATE CASCADE ON DELETE SET NULL");
        }
        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {
            modelBuilder.Entity<TeamMeber>()
                .HasOptional(tm => tm.Team)
                .WithMany(t => t.TeamMembers).Map(m=>m.MapKey())
                .WillCascadeOnDelete(false);
            base.OnModelCreating(modelBuilder);
        }

    }
}
