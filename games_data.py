# Used for gathering games data from storage
import os
import json
import csv

import steam_api


class GamesData:
    def __init__(self, dir_users, dir_games, file_ledger, file_users_conv, time_value_function, use_recent, min_games_played):
        self.dir_users = dir_users
        self.dir_games = dir_games
        self.file_ledger = file_ledger
        self.file_users_conv = file_users_conv
        self.min_games_played = min_games_played

        if time_value_function is None:
            self.time_value_function = lambda x: min(x / 30000.0, 1)
        else:
            self.time_value_function = time_value_function

        self.use_recent = use_recent

        print('\tLoading Ledger')
        self.ledger_to, self.ledger_from = self.__get_ledger()
        print('\tLoading Original Games Mapping')
        self.games_map = self.__map_orig_games()
        self.games_count = len(self.ledger_to)
        print('\tConverting Users')
        self.users_conv = self.__get_users_conv()

    def __map_orig_games(self):
        gmap = {}
        for f in os.listdir(self.dir_games):
            game = steam_api.Game(json.load(open(os.path.join(self.dir_games, f), "r+")))
            gmap[game.appid] = game
        return gmap

    # Creates a 1 to 1 mapping for all game objects to a 0 based index
    def __make_ledger(self):
        i = 0
        ledger_to = {}
        for f in os.listdir(self.dir_games):
            game = steam_api.Game(json.load(open(os.path.join(self.dir_games, f), "r+")))
            ledger_to[game.appid] = i
            i += 1

        json.dump(ledger_to, open(self.file_ledger, 'w+'))
        return ledger_to

    def __load_ledger(self):
        ledger_to = json.load(open(self.file_ledger, "r+"))
        return ledger_to

    def __invert_ledger(self, ledger_to):
        ledger_from = {}
        for (k, v) in ledger_to.items():
            ledger_from[v] = k
        return ledger_from

    def __get_ledger(self):
        ledger_to = self.__load_ledger() if os.path.exists(self.file_ledger) else self.__make_ledger()
        return ledger_to, self.__invert_ledger(ledger_to)

    def __count_games_played(self, games):
        count = 0
        for g in games:
            if ((g['playtime_2weeks'] if 'playtime_2weeks' in g else 0)
                    if self.use_recent else g['playtime_forever']) > 0:
                count += 1
        return count

    def __make_users_conv(self):
        fi = open(self.file_users_conv, 'w+', newline='')
        stream_out = csv.writer(fi)
        for f in os.listdir(self.dir_users):
            dat = json.load(open(os.path.join(self.dir_users, f), 'r+'))
            usr_games = [0] * self.games_count
            if self.__count_games_played(dat['games'].values()) > self.min_games_played:
                for (appid, g) in dat['games'].items():
                    usr_games[self.ledger_to[appid]] = self.time_value_function(
                        (g['playtime_2weeks'] if 'playtime_2weeks' in g else 0)
                        if self.use_recent else g['playtime_forever'])
                stream_out.writerow(usr_games)
            del dat

        # json.dump(users_conv, open(self.file_users_conv, 'w+'))
        fi.close()
        # return users_conv

    # def __load_users_conv(self):
    #    return json.load(open(self.file_users_conv, 'r+'))

    def __get_users_conv(self):
        # return self.__load_users_conv() if os.path.exists(self.file_users_conv) else self.__make_users_conv()
        if not os.path.exists(self.file_users_conv):
            self.__make_users_conv()
        f = open(self.file_users_conv, 'r+')
        return f, csv.reader(f, delimiter=',', lineterminator='\r\n')

    def get_game_from_index(self, index):
        return self.games_map[self.ledger_from[index]]

    def get_users_conv(self):
        return self.users_conv


if __name__ == '__main__':
    gamedata = GamesData('./users', './games', 'ledger.json', 'users_conv.json', None, True)
    print('Loaded ' + str(len(gamedata.get_users_conv())) + ' users.')
