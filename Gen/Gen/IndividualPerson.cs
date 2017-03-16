using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace Gen
{
    public class IndividualPerson
    {
        public static Random _random = new Random();
        public List<int> Individual { get; set; }
        public int Mark { get; set; } = 0;
        public IndividualPerson(int size,int[] template)
        {
            GenerateInvidual(size);
            Mark = CountMarks(template);
        }
        public int CountMarks(int[] template)
        {
            var result = 0;
            for (var i = 0;i < Individual.Count;i++)
                if (Individual.Skip(i).First() == template[i])
                    result++;
            return result;
        }
        public void GenerateInvidual(int size)
        {
            Individual = new List<int>();
            for (var i = 0;i < size;i++)
            {
                var number = _random.Next(0,2);
                Individual.Add(number);
            }
        }
        public override string ToString()
        {
            var st = new StringBuilder();
            foreach (var item in Individual)
                st.Append(item + " ");
            st.Append(Mark);
            return st.ToString();
        }
    }
}
