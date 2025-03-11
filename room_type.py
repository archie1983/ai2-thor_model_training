from enum import Enum

###
# We need a unified Enum for room types.
###
class RoomType(Enum):
    LIVING_ROOM = 1
    KITCHEN = 2
    BEDROOM = 3
    BATHROOM = 4
    NOT_KNOWN = 5
    NOT_CLASSIFIED = 6
    OFFICE = 7
    STORAGE = 8

    @staticmethod
    def colour_of_room(room_type):
        if room_type == RoomType.LIVING_ROOM:
            return "#8F0000"
        elif room_type == RoomType.KITCHEN:
            return "#008F00"
        elif room_type == RoomType.BEDROOM:
            return "#00008F"
        elif room_type == RoomType.BATHROOM:
            return "#8F8F00"
        elif room_type == RoomType.OFFICE:
            return "#8F008F"
        elif room_type == RoomType.STORAGE:
            return "#008F8F"
        else:
            return "#8F8F8F"

        #match room_type:
        #    case RoomType.LIVING_ROOM:
        #        return "#8F0000"
        #    case RoomType.KITCHEN:
        #        return "#008F00"
        #    case RoomType.BEDROOM:
        #        return "#00008F"
        #    case RoomType.BATHROOM:
        #        return "#8F8F00"
        #    case RoomType.OFFICE:
        #        return "#8F008F"
        #    case RoomType.STORAGE:
        #        return "#008F8F"
        #    case _:
        #        return "#8F8F8F"

    @classmethod
    def parse_llm_response(self, text, skip_chars=0, include_office_and_storage = True):
        ret_val = RoomType.NOT_KNOWN
        nearest_index = 1000000

        if (skip_chars > 0):
            text = text[skip_chars:]

        ##
        # Find the first occurence of any of the rooms. This may not be perfect, but probably will do for now
        ##
        if "LIVING ROOM" in text.upper() and nearest_index > text.upper().find("LIVING ROOM"):
            ret_val = RoomType.LIVING_ROOM
            nearest_index = text.upper().find("LIVING ROOM")
        if "KITCHEN" in text.upper() and nearest_index > text.upper().find("KITCHEN"):
            ret_val = RoomType.KITCHEN
            nearest_index = text.upper().find("KITCHEN")
        if "BEDROOM" in text.upper() and nearest_index > text.upper().find("BEDROOM"):
            ret_val = RoomType.BEDROOM
            nearest_index = text.upper().find("BEDROOM")
        if "BATHROOM" in text.upper() and nearest_index > text.upper().find("BATHROOM"):
            ret_val = RoomType.BATHROOM
            nearest_index = text.upper().find("BATHROOM")
        if include_office_and_storage and "OFFICE" in text.upper() and nearest_index > text.upper().find("OFFICE"):
            ret_val = RoomType.OFFICE
            nearest_index = text.upper().find("OFFICE")
        if include_office_and_storage and "STORAGE" in text.upper() and nearest_index > text.upper().find("STORAGE"):
            ret_val = RoomType.STORAGE
            nearest_index = text.upper().find("STORAGE")

        return ret_val

    @classmethod
    def interpret_label(self, text):
        ret_val = RoomType.NOT_KNOWN
        if "LIVING ROOM" in text.upper():
            ret_val = RoomType.LIVING_ROOM
        if "LIVINGROOM" in text.upper():
            ret_val = RoomType.LIVING_ROOM
        if "KITCHEN" in text.upper():
            ret_val = RoomType.KITCHEN
        if "BEDROOM" in text.upper():
            ret_val = RoomType.BEDROOM
        if "BATHROOM" in text.upper():
            ret_val = RoomType.BATHROOM
        if "OFFICE" in text.upper():
            ret_val = RoomType.OFFICE
        if "STORAGE" in text.upper():
            ret_val = RoomType.STORAGE

        return ret_val

    @classmethod
    def all_labels(self):
        return ["LIVING ROOM", "KITCHEN", "BEDROOM", "BATHROOM", "OFFICE", "STORAGE"]

    @classmethod
    def all_options(self, include_office_and_storage = True):
        if (include_office_and_storage):
            return [RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BEDROOM, RoomType.BATHROOM, RoomType.OFFICE, RoomType.STORAGE]
        else:
            return [RoomType.LIVING_ROOM, RoomType.KITCHEN, RoomType.BEDROOM, RoomType.BATHROOM]
