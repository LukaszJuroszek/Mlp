using System;
using System.Linq;
using EventsDal.Model;
using System.Xml;
using System.Collections.Generic;

namespace EventsDal.Concrete
{
    [Flags]
    public enum EventEnum
    {
        Id = 0,
        Title = 1,
        Description = 2,
        When = 3
    }
    public class XMLEventRepository : IEventRepository
    {
        private  static string  fileName= "Event.xml";
        public void Add(Event e)
        {
            var test = new XmlDocument();
            var node = test.CreateElement("Event");
            test.Save(fileName);
        }
        public void Delete(int id)
        {
            throw new NotImplementedException();
        }
        public void Edit(Event e)
        {
            throw new NotImplementedException();
        }
        public IQueryable<Event> GetAll()
        {
            var results = new List<Event>();
            var document = new XmlDocument();
            document.Load(fileName);
            foreach (XmlNode item in document.DocumentElement)
            {
                results.Add(new Event
                {
                    Id = int.Parse(item.Attributes[(int)EventEnum.Id].Value),
                    Title = item.ChildNodes[0].ChildNodes[(int)EventEnum.Title].Value,
                    Description = item.ChildNodes[0].ChildNodes[(int)EventEnum.Description].Value,
                    When = DateTime.Parse(item.ChildNodes[0].ChildNodes[(int)EventEnum.When].Value)
                });
            }
            return results.AsQueryable();

        }
        public Event GetByID(int id)
        {
            throw new NotImplementedException();
        }
    }
}
