using System;
using System.Collections.Generic;
using T4.Model;

namespace T4
{
    class Program
    {
        static void InitData(CompanyContext context)
        {
            context.Team.Add(new Team
            {
                Name = "ATH",
                TeamMembers = new List<TeamMeber>
                {
                    new TeamMeber{Name="Janek",MemberType=MemberType.Developer},
                    new TeamMeber{Name="Andrzej",MemberType=MemberType.ProduktOwner},
                    new TeamMeber{Name="Grzegorz",MemberType=MemberType.Developer}
                }
            });
            context.Team.Add(new Team
            {
                Name = "ATH 2",
                TeamMembers = new List<TeamMeber>
                {
                    new TeamMeber{Name="Malgosia",MemberType=MemberType.Developer},
                    new TeamMeber{Name="Zoisa",MemberType=MemberType.ProduktOwner},
                    new TeamMeber{Name="Marta",MemberType=MemberType.Developer}
                }
            });
            context.SaveChanges();
        }
        //codefirst Entiti
        static void Main(string[] args)
        {
            var dataPath = AppDomain.CurrentDomain.BaseDirectory;
            AppDomain.CurrentDomain.SetData("DataDirectory", dataPath);
            using (var db = new CompanyContext())
            {
                db.Database.CreateIfNotExists();
            }
        }
    }
}
