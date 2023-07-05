/*test包含测试函数，即set指令以及dump输出指令*/
#include"test.h"
#include "map.h"
int setMoney(char*arg2,char*arg3)
{

}
int setPoint(char*arg2,char*arg3)
{

}
int setItem(char*arg2,char*arg3,char*arg4)
{

}
int setBuff(char*arg2,char*arg3)
{

}
int setMap(char*arg2,char*arg3,char*arg4)
{

}
int setUnmap(char*arg2)
{

}
int setBomb(Player*player,char*arg2)
{

}
int setBarrier(Player*player,char*arg2)
{

}
int setPos(char*arg2,char*arg3)
{

}
int setQuit(char*arg2)
{

}
int setStop(char*arg2,char*arg3)
{
    
}
/*
    function:dump one player's related info into the file
    parameter : player info to be added
    return value : 1: error occured, 0:successfully performed
*/
int Dump_One_Player(Player *p){
	Dump_Line(p->name,0);
    char line[50];
    sprintf(line, "alive [%d]\n",p->alive );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    sprintf(line, "money [%d]",p->fund );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    sprintf(line, "point [%d]",p->points );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    sprintf(line, "item1 [%d]",p->toolnum[1] );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));

    sprintf(line, "item2 [%d]",p->toolnum[2] );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));

    sprintf(line, "item3 [%d]",p->toolnum[3] );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));

    sprintf(line, "buff [%d]",p->buff );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));

    sprintf(line, "stop [%d]",p->stop );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));

    sprintf(line, "userloc [%d]",p->loc );
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    return 1;

}

/*

*/
int Dump_Line(char * line,int line_num){
    char file_path[20] = "../dump.txt"; 
    FILE* fp = fopen(file_path,"a+");
    if (!fp)
    {
        printf("error occured during an attempt to open dump.txt");
        return 1;//failed
    }
    fprintf(fp, "%s\n",line);;
    fclose(fp);
    return 0;
}
int Dump_line_overload();

int dump_file(Player* p){
	char line[50];//single line of dump.txt file
    char P_Order[5] = "QASJ";//for test use
    sprintf(line,"user [%c|%c|%c|%c]",P_Order[0],P_Order[1],P_Order[2],P_Order[3]);
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    sprintf(line,"preuser [%s]",p->name);
    Dump_Line(line,0);
    memset(line,0,sizeof(line));
    for(int i = 0;i<4;i++){
        Dump_One_Player(p);
        p = p->next;
    }
    Dump_Line("MAP",0);

    for (int i = 0; i < 70; ++i)
    {
        sprintf(line,"mapuser [%d] [%s]",i,map[i].user);
        Dump_Line(line,0);
    }
    for (int i = 0; i < 70; ++i)
    {
        sprintf(line,"map [%d] [%d] [%d]",i,map[i].whom,map[i].level);
        Dump_Line(line,0);
    }
    for (int i = 0; i < 70; ++i)
    {
        if(map[i].item[0]|map[i].item[2] == 0) continue;
        int item_t = map[i].item[0] > 0 ? 1:3;
        sprintf(line,"item [%d] [%d]",i,item_t);
        Dump_Line(line,0);
    }
    return 0;

}
