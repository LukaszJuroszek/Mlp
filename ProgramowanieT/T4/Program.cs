using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
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
                db.Database.Log = Console.WriteLine;
                //delete with select
                //var toRemove = db.TeamMembers.FirstOrDefault(n => n.Id == 1);
                //db.TeamMembers.Remove(toRemove);
                //db.SaveChanges();
                //delete without select or 
                //var toRemove = new TeamMeber { Id = 2 };
                //db.TeamMembers.Add(toRemove);
                //db.TeamMembers.Remove(toRemove);
                //db.SaveChanges();
                //or
                //var toRemove = new TeamMeber { Id = 2 };
                //db.Entry(toRemove).State = EntityState.Deleted;
                //db.SaveChanges();
                //add
                //InitData(db);
                //db.Database.CreateIfNotExists();
                //var members = (from n in db.TeamMembers.Include(n => n.Team) select n);
                //foreach (var item in members)
                //{
                //    Console.WriteLine($"Mebers: {item.Name}, Team: {item.Team.Name}");
                //}
            }
        }
    }
}
