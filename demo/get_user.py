import socket
import getpass
rank = 0
user_name = getpass.getuser()  # 获取当前用户名
hostname = socket.gethostname()  # 获取当前主机名
ip_address = socket.gethostbyname(hostname)
# if hostname == "mgmt-1":
#     rank = 0
# elif hostname == "exec-1":
#     rank = 1
# elif hostname == 'exec-2':
#     rank = 2
# else:
#     rank = 3

print(type(user_name))
print(user_name)
print(hostname)
print(ip_address)

# mgmt-1 192.168.101.40
# exec-1 192.168.101.41
# exec-2 192.168.101.42
# exec-3 192.168.101.43
