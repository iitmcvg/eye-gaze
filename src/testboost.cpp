
#include <boost/mpi.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>

int main(int argc, char* argv[])
{
  mpi::environment env(argc, argv);
  mpi::communicator world;

  std::string msg;
  world.recv(0, 0, msg);
  std::cout << msg << ", ";
  std::cout.flush();
  world.send(0, 1, std::string("world"));
  
  return 0;
}