using Dapper;
using EventsDal.Model;
using System.Configuration;
using System.Data.SqlClient;
using System.Data.SqlTypes;
using System.Linq;

namespace EventsDal.Concrete
{
    public class SQLEventRepository : IEventRepository
    {
        private const string stringFromat = "yyyy-MM-dd HH:mm:ss";
        private const string stringFromat2 = "yyyy-MM-dd";

        private static SqlConnection DBConnection() => new SqlConnection(ConfigurationManager.ConnectionStrings["EventsDapper"].ConnectionString);
        private bool RowsAffected(int rowsAffected) => rowsAffected > 0 ? true : false;
        public void Add(Event e)
        {
            using (var db = DBConnection())
            {
                db.Open();
                var rowsAffected = db.Execute("INSERT INTO Event values(@Title,@Description,@When)",new { Title = e.Title,Description = e.Description,When = e.When.Date.ToString(stringFromat) });
                //return RowsAffected(rowsAffected);
            }
        }
        public void Delete(int id)
        {
            using (var db = DBConnection())
            {
                db.Open();
                var rowsAffected = db.Execute("DELETE FROM Event WHERE Id = @Id",new { Id = id });
                //  return RowsAffected(rowsAffected);
            }
        }
        public void Edit(Event e)
        {
            using (var db = DBConnection())
            {
                db.Open();
                //SQL WHEN is slq syntax !!!
                var rowsAffected = db.Query("UPDATE Event SET Title = @Title ,Description = @Description, \"When\" = @When WHERE Id =  @Id",new { Id = e.Id,Title = e.Title,Description = e.Description,When = e.When.Date.ToString(stringFromat2) });
                //return RowsAffected(rowsAffected);
            }
        }
        public IQueryable<Event> GetAll()
        {
            using (var db = DBConnection())
            {
                db.Open();
                var result = db.Query<Event>("SELECT TOP 1000 * FROM Event");
                return result.AsQueryable();
            }
        }
        public Event GetByID(int id)
        {
            using (var db = DBConnection())
            {
                db.Open();
                var result = db.QueryFirstOrDefault<Event>("SELECT * FROM Event WHERE Id = @Id",new { Id = id });
                return result;
            }
        }
    }
}
