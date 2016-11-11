//============================================================================//
// A simple parser class to read text file                                    //
//                                                                            //
// Chao Peng                                                                  //
// 06/07/2016                                                                 //
//============================================================================//

#include "ConfigParser.h"
#include <streambuf>
#include <cstring>
#include <climits>
#include <algorithm>
#include <list>

using namespace std;



//============================================================================//
// Constructor, Destructor                                                    //
//============================================================================//
ConfigParser::ConfigParser(const string &s,
                           const string &w,
                           const vector<string> &c)
: splitters(s), white_space(w), comment_marks(c), line_number(0),
  in_comment_pair(false)
{
    comment_pair = std::make_pair("/*", "*/");
}

ConfigParser::~ConfigParser()
{
    // place holder
}

//============================================================================//
// Public Member Function                                                     //
//============================================================================//

// add comment mark
void ConfigParser::AddCommentMark(const string &c)
{
    auto it = find(comment_marks.begin(), comment_marks.end(), c);
    if(it == comment_marks.end())
        comment_marks.push_back(c);
}

// remove certain comment mark
void ConfigParser::RemoveCommentMark(const string &c)
{
    auto it = find(comment_marks.begin(), comment_marks.end(), c);
    if(it != comment_marks.end())
        comment_marks.erase(it);
}

// clear all comment marks
void ConfigParser::EraseCommentMarks()
{
    comment_marks.clear();
}

// open a file for future parsing
bool ConfigParser::OpenFile(const string &path)
{
    Clear();

    infile.open(path);

    return infile.is_open();
}

// read the whole file into a buffer and break it into lines
bool ConfigParser::ReadFile(const string &path)
{
    Clear();

    ifstream f(path);

    if(!f.is_open())
        return false;

    string buffer;

    f.seekg(0, ios::end);
    buffer.reserve(f.tellg());
    f.seekg(0, ios::beg);

    buffer.assign((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    f.close();

    // remove comments and break the buffers into lines
    buffer_process(buffer);
    return true;
}

// read a buffer and break it into lines
void ConfigParser::ReadBuffer(const char *buf)
{
    Clear();

    string buffer = buf;

    // remove comments and break the buffers into lines
    buffer_process(buffer);
}

// clear stored lines
void ConfigParser::Clear()
{
    line_number = 0;
    infile.close();
    queue<string>().swap(lines);
}

// close file
void ConfigParser::CloseFile()
{
    infile.close();
}

// take a line
string ConfigParser::TakeLine()
{
    if(lines.size()) {
        string out = lines.front();
        lines.pop();
        return out;
    }

    return "";
}

// parse a line, take the line either from the opening file (if it is opened)
// or from the stored lines
// return false if empty
bool ConfigParser::ParseLine()
{
    queue<string>().swap(elements);

    if(infile.is_open())
        return parse_file();
    else
        return parse_buffer();
}

// parse a input string (line)
// trim the white space and split the string into elements by defined splitters
// the elements will be further trimmed
void ConfigParser::ParseLine(const string &line, const bool &count)
{
    if(count)
        ++line_number;

    string trim_line = trim(comment_out(line), white_space);

    queue<string> eles = split(trim_line, splitters);

    while(eles.size())
    {
        string ele = trim(eles.front(), white_space);
        if(ele.size())
            elements.push(ele);
        eles.pop();
    }
}

bool ConfigParser::CheckElements(int num)
{
    if(elements.size() == (size_t)num)
        return true;

    std::cout << "Config Parser Warning: Wrong format at line " << line_number
              << ", expecting " << num << " elements. "
              << std::endl
              << "\"" << current_line << "\""
              << std::endl;
    return false;
}


// take the first element
ConfigValue ConfigParser::TakeFirst()
{
    if(elements.empty()) {
        cout << "Config Parser: WARNING, trying to take elements while there is nothing, 0 value returned." << endl;
        return ConfigValue("0");
    }

    string output = elements.front();
    elements.pop();

    return ConfigValue(output);
}



//============================================================================//
// Private Member Function                                                    //
//============================================================================//

// process the read-in buffer
void ConfigParser::buffer_process(string &buffer)
{
    if(comment_pair.first.size() && comment_pair.second.size()) {
        // comment by pair marks
        while(comment_between(buffer, comment_pair.first, comment_pair.second))
        {;}
    }

    string line;
    for(auto &c : buffer)
    {
        if( c != '\n') {
            line += c;
        } else {
            lines.push(line);
            line.clear();
        }
    }
}

// take a line from opened file and parse it
// comment with comment pair will be removed here
inline bool ConfigParser::parse_file()
{
    // comment pair needs to be taken care here
    while(elements.empty())
    {
        if(!getline(infile, current_line))
            return false; // end of file

        // no need to take care comment pair
        if(comment_pair.first.empty() || comment_pair.second.empty()) {
            ParseLine(current_line);
        } else {
            // if we had comment pair opened
            if(in_comment_pair) {
                // see if there is an end to the comment pair
                auto c_end = current_line.find(comment_pair.second);
                if(c_end != string::npos) {
                    // end of a comment pair
                    ParseLine(current_line.substr(c_end + comment_pair.second.size()));
                    in_comment_pair = false;
                } else {
                    // still inside a comment pair
                    ParseLine("");
                }
            // if no previous comment pair opened
            } else {
                // remove complete comment pair in one line
                while(comment_between(current_line, comment_pair.first, comment_pair.second))
                {;}
                // see if there is any comment pair opening
                auto c_beg = current_line.find(comment_pair.first);
                // find comment pair openning
                if(c_beg != string::npos) {
                    ParseLine(current_line.substr(0, c_beg));
                    in_comment_pair = true;
                } else {
                    // no special treatment
                    ParseLine(current_line);
                }
            }
        }
    }

    return true;
}

// take a line from the stored lines buffer and parse it
// comment by comment pair has been already removed before going into the lines
inline bool ConfigParser::parse_buffer()
{
    // comment pair has been taken care before breaking into lines,
    // so it's simple here
    while(elements.empty())
    {
        if(lines.empty())
            return false; // end of buffer
        current_line = move(lines.front());
        lines.pop();
        ParseLine(current_line);
    }

    return true; // parsed a line
}

// comment out between pairs
// return false if no pair found
bool ConfigParser::comment_between(string &str, const string &open, const string &close)
{
    auto begin = str.find(open);
    if(begin != string::npos) {
        auto end = str.find(close, begin + open.size());
        if(end != string::npos) {
            str.erase(begin, end + close.size() - begin);
            return true;
        }
    }

    return false;
}

// comment out the characters with certain mark
string ConfigParser::comment_out(const string &str, size_t index)
{
    if(index >= comment_marks.size())
        return str;
    else {
        string str_co = comment_out(str, comment_marks.at(index));
        return comment_out(str_co, ++index);
    }
}



//============================================================================//
// Public Static Function                                                     //
//============================================================================//

// comment out the characters after comment mark
string ConfigParser::comment_out(const string &str, const string &c)
{
    if(c.empty())
        return str;

    const auto commentBegin = str.find(c);
    return str.substr(0, commentBegin);
}

// trim all the characters defined as white space at both ends
string ConfigParser::trim(const string &str, const string &w)
{

    const auto strBegin = str.find_first_not_of(w);
    if (strBegin == string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(w);

    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

// split a string into several pieces by all the characters defined as splitter
queue<string> ConfigParser::split(const string &str, const string &s)
{
    queue<string> eles;

    char *cstr = new char[str.length() + 1];

    strcpy(cstr, str.c_str());

    char *pch = strtok(&cstr[0], s.c_str());
    string ele;

    while(pch != nullptr)
    {
        ele = pch;
        eles.push(pch);
        pch = strtok(nullptr, s.c_str());
    }

    delete cstr;

    return eles;
}

// find the integer in a string
int ConfigParser::find_integer(const string &str, const size_t &pos)
{
    vector<int> integers = find_integers(str);
    if(pos >= integers.size())
    {
        cerr << "Config Parser: Cannot find " << pos + 1 << " integers from "
             << "\"" << str << "\"."
             << endl;
        return 0;
    }

    return integers.at(pos);
}

// find all the integers in a string
vector<int> ConfigParser::find_integers(const string &str)
{
    vector<int> result;

    find_integer_helper(str, result);

    return result;
}

// helper function for finding a integer
void ConfigParser::find_integer_helper(const string &str, vector<int> &result)
{
   if(str.empty())
       return;

   int negative = 1;
   auto numBeg = str.find_first_of("-0123456789");
   if(numBeg == string::npos)
       return;

   // check negative sign
   string str2 = str.substr(numBeg);

   if(str2.at(0) == '-')
   {
       negative = -1;
       int num_check;

       do {
           str2.erase(0, 1);

           if(str2.empty())
               return;

           num_check = str2.at(0) - '0';
       } while (num_check > 9 || num_check < 0);
   }

   auto numEnd = str2.find_first_not_of("0123456789");
   if(numEnd == string::npos)
       numEnd = str2.size();

   int num = 0;
   size_t i = 0;

   for(; i < numEnd; ++i)
   {
       if( (num > INT_MAX/10) ||
           (num == INT_MAX/10 && ((str2.at(i) - '0') > (INT_MAX - num*10))) )
       {
           ++i;
           break;
       }

       num = num*10 + str2.at(i) - '0';
   }

   result.push_back(negative*num);
   find_integer_helper(str2.substr(i), result);
}

// return the lower case of this string
string ConfigParser::str_lower(const string &str)
{
    string res = str;
    for(auto &c : res)
    {
        c = tolower(c);
    }
    return res;
}

// return the upper case of this string
string ConfigParser::str_upper(const string &str)
{
    string res = str;
    for(auto &c : res)
    {
        c = toupper(c);
    }
    return res;
}

// remove characters in ignore list
string ConfigParser::str_remove(const string &str, const string &ignore)
{
    string res = str;

    for(auto &c : ignore)
    {
        res.erase(remove(res.begin(), res.end(), c), res.end());
    }
    return res;
}

// replace characters in the list with certain char
string ConfigParser::str_replace(const string &str, const string &list, const char &rc)
{
    if(list.empty())
        return str;

    string res = str;

    for(auto &c : res)
    {
        if(list.find(c) != string::npos)
            c = rc;
    }

    return res;
}

// compare two strings, can be case insensitive
bool ConfigParser::strcmp_case_insensitive(const string &str1, const string &str2)
{
    if(str1.size() != str2.size()) {
        return false;
    }

    for(auto c1 = str1.begin(), c2 = str2.begin(); c1 != str1.end(); ++c1, ++c2)
    {
        if(tolower(*c1) != tolower(*c2)) {
            return false;
        }
    }

    return true;
}

// find pair position in a string
vector<pair<int, int>> ConfigParser::find_pair(const string &str,
                                               const string &op,
                                               const string &cl)
{
    vector<pair<int, int>> result;

    if(op == cl) {
        cerr << "find_between: do not accept identical open and close marks."
                  << endl;
        return result;
    }

    if(op.empty() || cl.empty())
        return result;

    int mark_size = op.size() + cl.size();

    // there won't be any
    if(mark_size > (int)str.size())
        return result;

    list<int> opens;
    for(int i = 0; i < (int)str.size(); ++i)
    {
        if(str.substr(i, op.size()) == op)
            opens.push_back(i);
    }

    list<int> closes;
    for(int i = 0; i < (int)str.size(); ++i)
    {
        if(str.substr(i, cl.size()) == cl)
        {
            bool share_char = false;
            for(auto &idx : opens)
            {
                if(abs(i - idx) < op.size())
                    share_char = true;
            }

            if(!share_char)
                closes.push_back(i);
        }
    }

    if(closes.size() != opens.size()) {
        cout << "find_between: find unmatched open and close marks."
                  << endl;
    }

    for(auto cl_it = closes.begin(); cl_it != closes.end(); ++cl_it)
    {
        for(auto op_it = opens.rbegin(); op_it != opens.rend(); ++op_it)
        {
            if((*cl_it - *op_it) >= mark_size)
            {
                result.emplace_back(*op_it, *cl_it);
                opens.erase(next(op_it).base());
                break;
            }
        }
    }

    return result;
}



//============================================================================//
// trivial operators                                                          //
//============================================================================//

ConfigParser &operator >> (ConfigParser &c, string &v)
{
    v = c.TakeFirst().String();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, bool &v)
{
    v = c.TakeFirst().Bool();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, char &v)
{
    v = c.TakeFirst().Char();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, unsigned char &v)
{
    v = c.TakeFirst().UChar();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, short &v)
{
    v = c.TakeFirst().Short();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, unsigned short &v)
{
    v = c.TakeFirst().UShort();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, int &v)
{
    v = c.TakeFirst().Int();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, unsigned int &v)
{
    v = c.TakeFirst().UInt();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, long &v)
{
    v = c.TakeFirst().Long();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, unsigned long &v)
{
    v = c.TakeFirst().ULong();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, long long &v)
{
    v = c.TakeFirst().LongLong();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, unsigned long long &v)
{
    v = c.TakeFirst().ULongLong();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, float &v)
{
    v = c.TakeFirst().Float();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, double &v)
{
    v = c.TakeFirst().Double();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, long double &v)
{
    v = c.TakeFirst().LongDouble();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, const char *&v)
{
    v = c.TakeFirst().c_str();
    return c;
}

ConfigParser &operator >> (ConfigParser &c, ConfigValue &v)
{
    v = c.TakeFirst();
    return c;
}
