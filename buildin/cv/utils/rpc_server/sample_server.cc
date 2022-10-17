/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 *************************************************************************/

#include <cstring>
#include <mm_remote.h>
#include <iostream>
#include <string>

void PrintHelp() {
  std::cout << "Usage: mm_rpc_server port\n";
}

int main() {
  magicmind::IRpcServer *server = magicmind::CreateRpcServer("8008");
  server->HandleRPCsLoop();
  server->Shutdown();
  return 0;
}

