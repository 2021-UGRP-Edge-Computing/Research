Commands
========

.. contents:: **목차**
    :local:
    

IP Address
----------

.. code:: bash

  hostname -I
  
..

  IP 주소를 얻을 수 있다. LAN 주소와 Wifi 주소가 다르니 유의한다.




Data Transmission
-----------------

.. code:: bash

  scp [파일명].[확장자] [수신자 이름]@[수신자 IP]:[directory/directory/...]
  scp text.txt icnl@192.168.0.10:/home/icnl

..
  
  최소한 /home/icnl 디렉토리까지 쳐야 한다.
