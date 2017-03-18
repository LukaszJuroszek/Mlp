using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EventsDal.Model
{
    public interface IEventRepository
    {
        void Add(Event e);
        void Edit(Event e);
        void Delete(int id);
        IQueryable<Event> GetAll();
        Event GetByID(int id);
    }
}
