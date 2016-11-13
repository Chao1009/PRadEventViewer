#ifndef CONFIG_VALUE_H
#define CONFIG_VALUE_H

#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <typeinfo>

// demangle type name
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
// gnu compiler needs to demangle type info
static std::string demangle(const char* name)
{

    int status = 0;

    //enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}
#else
// do nothing if not gnu compiler
static std::string demangle(const char* name)
{
    return name;
}
#endif

// this helps template specialization in class
template <typename T>
struct __cv_identifier { typedef T type; };

class ConfigValue
{
public:
    friend class ConfigParser;
    friend class ConfigObject;

public:
    ConfigValue() {};

    ConfigValue(const std::string &value);
    ConfigValue(std::string &&value);
    ConfigValue(const bool &value);
    ConfigValue(const int &value);
    ConfigValue(const long &value);
    ConfigValue(const long long &value);
    ConfigValue(const unsigned &value);
    ConfigValue(const unsigned long &value);
    ConfigValue(const unsigned long long &value);
    ConfigValue(const float &value);
    ConfigValue(const double &value);
    ConfigValue(const long double &value);

    ConfigValue &operator =(const std::string &str);
    ConfigValue &operator =(std::string &&str);

    bool Bool() const;
    char Char() const;
    unsigned char UChar() const;
    short Short() const;
    unsigned short UShort() const;
    int Int() const;
    unsigned int UInt() const;
    long Long() const;
    long long LongLong() const;
    unsigned long ULong() const;
    unsigned long long ULongLong() const;
    float Float() const;
    double Double() const;
    long double LongDouble() const;
    const char *c_str() const;
    const std::string &String() const {return _value;};
    bool IsEmpty() const {return _value.empty();};

    operator std::string()
    const
    {
        return _value;
    };

    bool operator ==(const std::string &rhs)
    const
    {
        return _value == rhs;
    }

    template<typename T>
    T Convert()
    const
    {
        return convert( __cv_identifier<T>());
    };

private:
    std::string _value;

    template<typename T>
    T convert(__cv_identifier<T> &&)
    const
    {
        std::stringstream iss(_value);
        T _cvalue;

        if(!(iss >> _cvalue)) {
            std::cerr << "Config Value Warning: Undefined value returned, failed to convert "
                      <<  _value
                      << " to "
                      << demangle(typeid(T).name())
                      << std::endl;
        }

        return _cvalue;
    }

    ConfigValue convert(__cv_identifier<ConfigValue>)
    const
    {
        return *this;
    }

    bool convert(__cv_identifier<bool> &&)
    const
    {
        return (*this).Bool();
    }

    std::string convert(__cv_identifier<std::string> &&)
    const
    {
        return (*this)._value;
    }

};

// show string content of the config value to ostream
std::ostream &operator << (std::ostream &os, const ConfigValue &b);

#endif
