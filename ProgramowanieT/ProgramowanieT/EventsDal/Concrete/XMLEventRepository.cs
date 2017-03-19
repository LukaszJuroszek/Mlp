using System;
using System.Linq;
using EventsDal.Model;
using System.Collections.Generic;
using System.Xml.Linq;

namespace EventsDal.Concrete
{
    public enum EventEnum
    {
        Events,
        Event,
        Id,
        Title,
        Description,
        When
    }
    public class XMLEventRepository : IEventRepository
    {
        private  static string  fileName= "Event.xml";
        public void Add(Event e)
        {
            var doc = XDocument.Load(fileName);
            XElement root = CreateXElementFromEvent(e);
            doc.Element(EventEnum.Events.ToString()).Add(root);
            doc.Save(fileName);
        }
        private static XElement CreateXElementFromEvent(Event e)
        {
            var root = new XElement(EventEnum.Event.ToString());
            root.Add(new XAttribute(EventEnum.Id.ToString(),e.Id));
            root.Add(new XElement(EventEnum.Title.ToString(),e.Title));
            root.Add(new XElement(EventEnum.Description.ToString(),e.Description));
            root.Add(new XElement(EventEnum.When.ToString(),e.When));
            return root;
        }
        public void Edit(Event e)
        {
            var doc = XDocument.Load(fileName);
            var xElementToEdit = GetEventXElementByIdInXDocument(e.Id,doc);
            xElementToEdit.Element(EventEnum.Title.ToString()).Value = e.Title;
            xElementToEdit.Element(EventEnum.Description.ToString()).Value = e.Description;
            xElementToEdit.Element(EventEnum.When.ToString()).Value = e.When.ToString();
            doc.Save(fileName);
        }
        public void Delete(int id)
        {
            var doc = XDocument.Load(fileName);
            GetEventXElementByIdInXDocument(id,doc)
                 .Remove();
            doc.Save(fileName);
        }
        public IQueryable<Event> GetAll()
        {
            var doc = XDocument.Load(fileName);
            var elemnts = GetAllEventsFromXDocument(doc);
            var results = new List<Event>();
            for (var i = 0;i < elemnts.Count();i++)
            {
                var item = elemnts.Skip(i).First();
                if (HasIdFristAtribute(item))
                    results.Add(GetEventFromXElement(item));
            }
            return results.AsQueryable();
        }
        private static IEnumerable<XElement> GetAllEventsFromXDocument(XDocument doc)
        {
            return doc.Document.Descendants().Where(x => x.Name == EventEnum.Event.ToString());
        }
        private static Event GetEventFromXElement(XElement item)
        {
            return new Event
            {
                Id = GetIdFromXElement(item),
                Title = GetTitleFromXElement(item),
                Description = GetDescriptionFromXElement(item),
                When = GetWhenFromXElement(item)
            };
        }
        private static DateTime GetWhenFromXElement(XElement item)
        {
            return DateTime.Parse(item.Element(EventEnum.When.ToString()).Value);
        }
        private static string GetDescriptionFromXElement(XElement item)
        {
            return item.Element(EventEnum.Description.ToString()).Value;
        }
        private static string GetTitleFromXElement(XElement item)
        {
            return item.Element(EventEnum.Title.ToString()).Value;
        }
        private static int GetIdFromXElement(XElement item)
        {
            return int.Parse(item.FirstAttribute.Value);
        }
        private static bool HasIdFristAtribute(XElement item)
        {
            return item.FirstAttribute.Name.LocalName.Equals(EventEnum.Id.ToString());
        }
        public Event GetByID(int id)
        {
            var doc = XDocument.Load(fileName);
            return GetEventFromXElement(GetEventXElementByIdInXDocument(id,doc));
        }
        private static XElement GetEventXElementByIdInXDocument(int id,XDocument doc)
        {
            return GetAllEventsFromXDocument(doc)
                            .Where(x => GetIdFromXElement(x) == id)
                            .First();
        }
    }
}
