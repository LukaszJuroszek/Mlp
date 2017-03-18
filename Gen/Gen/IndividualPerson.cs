using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace Gen
{
    public class IndividualPerson
    {
        public static Random _random = new Random();
        private List<int> individual;
        public int Mark { get; set; } = 0;
        public List<int> GetIndividual()
        {
            return individual;
        }
        public void SetIndividual(List<int> value)
        {
            individual = value;
        }
        public void SetIndividual(List<int> value,int[] template)
        {
            individual = value;
            CountMarks(template);
        }
        public IndividualPerson(int[] template,int sizeOfeInvidual)
        {
            GenerateInvidual(sizeOfeInvidual);
            CountMarks(template);
        }
        public int GetIndividualSize() => GetIndividual().Count;
        public void CountMarks(int[] template)
        {
            var result = 0;
            for (var i = 0;i < GetIndividual().Count;i++)
                if (GetIndividual()[i] == template[i])
                    result++;
            Mark = result;
        }
        public void GenerateInvidual(int size)
        {
            SetIndividual(new List<int>());
            for (var i = 0;i < size;i++)
            {
                var number = _random.Next(0,2);
                GetIndividual().Add(number);
            }
        }
        public override string ToString()
        {
            var st = new StringBuilder();
            foreach (var item in GetIndividual())
                st.Append(item + " ");
            st.Append($"Mark: {Mark}");
            return st.ToString();
        }
    }
}
