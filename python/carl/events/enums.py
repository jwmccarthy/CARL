from enum import IntEnum, IntFlag

from .. import _carl


class Event(IntFlag):
    BALL_TOUCH           = _carl.EVENT_BALL_TOUCH
    JUMP                 = _carl.EVENT_JUMP
    DOUBLE_JUMP          = _carl.EVENT_DOUBLE_JUMP
    DODGE                = _carl.EVENT_DODGE
    CONTACT_SURFACE      = _carl.EVENT_CONTACT_SURFACE
    FLIP_RESET           = _carl.EVENT_FLIP_RESET
    DEMOED               = _carl.EVENT_DEMOED
    GOAL                 = _carl.EVENT_GOAL
    CEILING_BODY_CONTACT = _carl.EVENT_CEILING_BODY_CONTACT
    LEAVE_SURFACE        = _carl.EVENT_LEAVE_SURFACE
    BALL_ACCELERATES     = _carl.EVENT_BALL_ACCELERATES


class ContactSurface(IntEnum):
    FRONT     = _carl.TOUCH_FRONT
    REAR      = _carl.TOUCH_REAR
    LEFT      = _carl.TOUCH_LEFT
    RIGHT     = _carl.TOUCH_RIGHT
    ROOF      = _carl.TOUCH_ROOF
    UNDERSIDE = _carl.TOUCH_UNDERSIDE


class ArenaSurface(IntFlag):
    FLOOR     = _carl.ARENA_FLOOR
    SIDE_WALL = _carl.ARENA_SIDE_WALL
    BACK_WALL = _carl.ARENA_BACK_WALL
    CEILING   = _carl.ARENA_CEILING
    OTHER     = _carl.ARENA_OTHER
