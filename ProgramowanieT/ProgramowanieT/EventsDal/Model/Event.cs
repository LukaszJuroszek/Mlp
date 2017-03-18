using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EventsDal.Model
{
    enum EventType
    {
        eConference,
        eMeeting,
        eCall
    }
    public class Event
    {
        public int Id { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public DateTime When { get; set; }
    }
}
