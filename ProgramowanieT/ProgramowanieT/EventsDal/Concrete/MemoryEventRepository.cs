using System;
using System.Linq;
using EventsDal.Model;
using System.Collections.Generic;

namespace EventsDal.Concrete
{
    public class MemoryEventRepository : IEventRepository
    {
        private List<Event> _events;
        public MemoryEventRepository()
        {
            _events = new List<Event> { new Event { Id = 1,Title = "ITAD",When = DateTime.Now.AddDays(4),Description=" " },
                new Event { Id = 2,Title = "Costam",When = DateTime.Now.AddDays(2),Description=" " } };
        }
        public void Add(Event e)
        {
            _events.Add(e);
        }
        public void Delete(int id)
        {
            _events.Remove(GetByID(id));
        }
        public void Edit(Event e)
        {
            var replacedItem = _events.Where(x => x.Id == e.Id).First();
            var indexOfItem = _events.IndexOf(replacedItem);
            if (indexOfItem!=-1)
            {
                _events[indexOfItem] = e;
            }
        }
        public IQueryable<Event> GetAll()
        {
            return _events.AsQueryable();
        }
        public Event GetByID(int id)
        {
            return _events.Where(x => x.Id == id).First();
        }
    }
}
