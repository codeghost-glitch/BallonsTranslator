
from enum import Enum
from typing import List, Optional, Sequence, Union, Tuple
import numpy as np
import cv2
import copy

from qtpy.QtWidgets import QApplication, QWidget
from qtpy.QtCore import QObject, QRectF, Qt, Signal, QPointF
from qtpy.QtGui import QKeyEvent, QTextCursor, QFontMetricsF, QFont, QTextCharFormat, QClipboard
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from ..item import TextBlkItem, TextBlock
from ...canvas import Canvas
from .widgets import TransTextEdit, SourceTextEdit, TransPairWidget, TextEditListScrollArea, QVBoxLayout, Widget
from ballontranslator.utils.fontformat import FontFormat
from .commands import (
    ApplyFontformatCommand,
    AutoLayoutCommand,
    CapitalizeTextItemsCommand,
    MoveBlkItemsCommand,
    MultiPasteCommand,
    PageReplaceAllCommand,
    PageReplaceOneCommand,
    ReshapeItemCommand,
    ResetAngleCommand,
    RotateItemCommand,
    SqueezeCommand,
    TextEditCommand,
    TextItemEditCommand,
    propagate_user_edit,
)
from ..formatting.panel import FontFormatPanel
from ballontranslator.utils.config import pcfg
from ballontranslator.utils import shared
from ballontranslator.utils.imgproc_utils import extract_ballon_region, get_block_mask
from ballontranslator.utils.text_processing import seg_text, is_cjk
from ballontranslator.utils.text_layout import (
    layout_text,
    hyphenate_long_words,
    row_width_profile,
)


def build_path_reorder_map(
    touched_ids: Sequence[int],
    item_count: int,
) -> Tuple[List[int], List[int]]:
    """Move touched items to the front in path order.

    >>> build_path_reorder_map([2, 0], 4)
    ([2, 0, 1], [0, 1, 2])
    """
    seen = set()
    order = []
    for item_id in touched_ids:
        if 0 <= item_id < item_count and item_id not in seen:
            seen.add(item_id)
            order.append(item_id)
    order.extend(item_id for item_id in range(item_count) if item_id not in seen)

    source_ids = []
    target_ids = []
    for target_id, source_id in enumerate(order):
        if source_id != target_id:
            source_ids.append(source_id)
            target_ids.append(target_id)
    return source_ids, target_ids


class SceneTextReplacementReason(Enum):
    """Describe why every scene text item is being replaced.

    ``PAGE_CHANGE`` assumes the caller resolved edits before saving the old
    page. The other reasons discard transient editors because their backing
    model has already been replaced externally.

    >>> SceneTextReplacementReason.CURRENT_PAGE_RELOAD.value
    'current-page-reload'
    """

    PAGE_CHANGE = 'page-change'
    CURRENT_PAGE_RELOAD = 'current-page-reload'
    PROJECT_RELOAD = 'project-reload'


class CreateItemCommand(QUndoCommand):
    def __init__(self, blk_item: TextBlkItem, ctrl, parent=None):
        super().__init__(parent)
        self.blk_item = blk_item
        self.ctrl: SceneTextManager = ctrl
        self.op_count = -1
        self.ctrl.addTextBlock(self.blk_item)
        self.pairw = self.ctrl.pairwidget_list[self.blk_item.idx]
        self.ctrl.txtblkShapeControl.setBlkItem(self.blk_item)

    def redo(self):
        if self.op_count < 0:
            self.op_count += 1
            self.blk_item.setSelected(True)
            return
        self.ctrl.recoverTextblkItemList([self.blk_item], [self.pairw])

    def undo(self):
        self.ctrl.deleteTextblkItemList([self.blk_item], [self.pairw])


class DeleteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], mode: int, ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = []
        self.pwidget_list: List[TransPairWidget] = []
        self.ctrl: SceneTextManager = ctrl
        self.sw = self.ctrl.canvas.search_widget
        self.canvas: Canvas = ctrl.canvas
        self.mode = mode

        self.undo_img_list = []
        self.redo_img_list = []
        self.inpaint_rect_lst = []
        self.mask_pnts = []
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        original_array = self.canvas.imgtrans_proj.img_array

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list = []
        self.highlighter_list = []
        self.old_counter_sum = self.sw.counter_sum
        self.sw_changed = False

        blk_list.sort(key=lambda blk: blk.idx)
        
        for blkitem in blk_list:
            if not isinstance(blkitem, TextBlkItem):
                continue
            self.blk_list.append(blkitem)
            pw: TransPairWidget = ctrl.pairwidget_list[blkitem.idx]
            self.pwidget_list.append(pw)

            if mode == 1:
                is_empty = False
                msk, xyxy = get_block_mask(blkitem.absBoundingRect(), mask_array, blkitem.rotation())
                if msk is None:
                    is_empty = True
                if is_empty:
                    self.undo_img_list.append(None)
                    self.redo_img_list.append(None)
                    self.inpaint_rect_lst.append(None)
                    self.mask_pnts.append(None)
                else:
                    x1, y1, x2, y2 = xyxy
                    self.mask_pnts.append(np.where(msk))
                    self.undo_img_list.append(np.copy(img_array[y1: y2, x1: x2]))
                    self.redo_img_list.append(np.copy(original_array[y1: y2, x1: x2]))
                    self.inpaint_rect_lst.append([x1, y1, x2, y2])

            rst_idx = self.sw.get_result_edit_index(pw.e_trans)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

            rst_idx = self.sw.get_result_edit_index(pw.e_source)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

        self.new_counter_sum = self.sw.counter_sum
        if self.sw_changed:
            if self.sw.counter_sum > 0:
                idx = self.sw.get_result_edit_index(self.sw.current_edit)
                if self.sw.current_cursor is not None and idx != -1:
                    self.sw.result_pos = self.sw.highlighter_list[idx].matched_map[self.sw.current_cursor.position()]
                    if idx > 0:
                        self.sw.result_pos += sum(self.sw.search_counter_list[: idx])
                    self.sw.updateCounterText()
                else:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
            else:
                self.sw.setCurrentEditor(None)

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)

    def redo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step -= 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, redo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.redo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = redo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 0
            self.canvas.updateLayers()

        if self.op_counter == 0:
            self.op_counter += 1
            return

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        if self.sw_changed:
            self.sw.counter_sum = self.new_counter_sum
            cursor_removed = False
            for edit in self.search_rstedit_list:
                idx = self.sw.get_result_edit_index(edit)
                if idx != -1:
                    self.sw.search_rstedit_list.pop(idx)
                    self.sw.search_counter_list.pop(idx)
                    self.sw.highlighter_list.pop(idx)
                if edit == self.sw.current_edit:
                    cursor_removed = True
            if cursor_removed:
                if self.sw.counter_sum > 0:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
                else:
                    self.sw.setCurrentEditor(None)

    def undo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step += 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, undo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.undo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = undo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 255
            self.canvas.updateLayers()

        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)
        if self.sw_changed:
            self.sw.counter_sum = self.old_counter_sum
            self.sw.search_rstedit_list += self.search_rstedit_list
            self.sw.search_counter_list += self.search_counter_list
            self.sw.highlighter_list += self.highlighter_list
            self.sw.updateCounterText()


class PasteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], pwidget_list: List[TransPairWidget], ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = blk_list
        self.ctrl:SceneTextManager = ctrl
        blk_list.sort(key=lambda blk: blk.idx)

        self.ctrl.canvas.block_selection_signal = True
        for blkitem in blk_list:
            blkitem.setSelected(True)
        self.ctrl.on_incanvas_selection_changed()
        self.ctrl.canvas.block_selection_signal = False
        self.pwidget_list = pwidget_list
        

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)

    def undo(self):
        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)


class PasteSrcItemsCommand(QUndoCommand):
    def __init__(self, src_list: List[SourceTextEdit], paste_list: List[str]):
        super().__init__()
        self.src_list = src_list
        self.paste_list = paste_list
        self.ori_text_list = [src.toPlainText() for src in src_list]

    def redo(self):
        for src, text in zip(self.src_list, self.paste_list):
            src.setPlainText(text)

    def undo(self):
        for src, text in zip(self.src_list, self.ori_text_list):
            src.setPlainText(text)


class RearrangeBlksCommand(QUndoCommand):

    def __init__(self, rmap: Tuple, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl: SceneTextManager = ctrl
        self.src_ids, self.tgt_ids = rmap[0], rmap[1]

        self.src2tgt = {}
        self.tgt2src = {}
        for s, t in zip(self.src_ids, self.tgt_ids):
            self.src2tgt[s] = t
            self.tgt2src[t] = s
        self.redo_visible_idx = self.undo_visible_idx = None
        if len(rmap) > 2:
            self.redo_visible_idx, self.undo_visible_idx = rmap[2]

    def redo(self):
        self.rearange_blk_ids(self.src_ids, self.tgt_ids, self.redo_visible_idx)

    def undo(self):
        self.rearange_blk_ids(self.tgt_ids, self.src_ids, self.undo_visible_idx)

    def rearange_blk_ids(self, src_ids, tgt_ids, visible_idx = None):
        src_ids = np.array(src_ids)
        tgt_ids = np.array(tgt_ids)
        src_order_ids = np.argsort(src_ids)[::-1]

        src_ids = src_ids[src_order_ids]
        tgt_ids = tgt_ids[src_order_ids]
        
        blks: List[TextBlkItem] = []
        pws: List[TransPairWidget] = []
        for pos, pos_tgt in zip(src_ids, tgt_ids):
            pw = self.ctrl.pairwidget_list.pop(pos)
            if visible_idx == pos_tgt:
                pw.hide()
            blk = self.ctrl.textblk_item_list.pop(pos)
            pws.append(pw)
            blks.append(blk)

        tgt_order_ids = np.argsort(tgt_ids)
        for ii in tgt_order_ids:
            pos = tgt_ids[ii]
            self.ctrl.textblk_item_list.insert(pos, blks[ii])
            
            self.ctrl.textEditList.insertPairWidget(pws[ii], pos)
            self.ctrl.pairwidget_list.insert(pos, pws[ii])

        self.ctrl.updateTextBlkItemIdx(set(tgt_ids))
        if visible_idx is not None:
            pw_ct = self.ctrl.pairwidget_list[visible_idx]
            pw_ct.show()
            self.ctrl.textEditList.ensureWidgetVisible(pw_ct, yMargin=pw.height())


class TextPanel(Widget):
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textEditList = TextEditListScrollArea(self)
        self.formatpanel = FontFormatPanel(app, self)
        layout.addWidget(self.formatpanel)
        layout.addWidget(self.textEditList)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


# App display names -> pyphen language tags. pcfg.module.translate_target
# holds display names ('English'), which pyphen rejects outright.
PYPHEN_LANGS = {
    'English': 'en',
    '简体中文': 'zh',
    '繁體中文': 'zh',
    '日本語': 'ja',
    '한국어': 'ko',
    'Tiếng Việt': 'vi',
    'čeština': 'cs',
    'Nederlands': 'nl',
    'Français': 'fr',
    'Deutsch': 'de',
    'magyar nyelv': 'hu',
    'Italiano': 'it',
    'Polski': 'pl',
    'Português': 'pt',
    'Brazilian Portuguese': 'pt',
    'limba română': 'ro',
    'русский язык': 'ru',
    'Español': 'es',
    'Türk dili': 'tr',
    'украї́нська мо́ва': 'uk',
    'Hindi': 'hi',
    'Malayalam': 'ml',
    'Tamil': 'ta',
}


def hyphenator_for_target(target: str):
    """pyphen hyphenator for an app display language name, or None.

    >>> hyphenator_for_target('English') is not None
    True
    >>> hyphenator_for_target('Unsupported Land') is None
    True
    """
    if not target or target not in PYPHEN_LANGS:
        return None
    try:
        import pyphen
        return pyphen.Pyphen(lang=pyphen.language_fallback(PYPHEN_LANGS[target]))
    except Exception:
        return None


def _attribution_points(blk) -> np.ndarray:
    """Line quads plus centroids for one block, block center as fallback.

    Shared by bubble attribution and outline ownership counting.
    """
    try:
        lines = np.asarray(blk.lines, np.float32).reshape(-1, 8)
    except (ValueError, TypeError):
        lines = np.empty((0, 8), np.float32)
    if lines.size:
        quads = lines.reshape(-1, 4, 2)
        return np.concatenate(
            [quads, quads.mean(axis=1, keepdims=True)], axis=1
        ).reshape(-1, 2)
    return np.asarray(blk.center(), np.float32).reshape(1, 2)


def _bubble_polygon_for(proj, blk) -> Optional[List]:
    """Page polygon of the detected bubble holding this block's text.

    Mirrors ProjImgTrans.prune_bubble_outlines attribution: line quads
    and their centroids first, block center as fallback; deepest
    containment wins when outlines nest.
    """
    outlines = proj.get_bubble_outlines(proj.current_img)
    if not outlines:
        return None
    pts = _attribution_points(blk)
    best = None
    best_depth = -1.0
    for poly in outlines:
        arr = np.asarray(poly, np.float32)
        for x, y in pts:
            # Measured distance, not the sign: sign-only comparison keeps
            # the first polygon that contains any point, which strands
            # seam-straddling blocks on a bubble they merely clip.
            depth = cv2.pointPolygonTest(arr, (float(x), float(y)), True)
            if depth > best_depth:
                best_depth = depth
                best = poly
    return best if best_depth >= 0 else None


class SceneTextManager(QObject):
    new_textblk = Signal(int)
    def __init__(self, 
                 app: QApplication,
                 mainwindow: QWidget,
                 canvas: Canvas, 
                 textpanel: TextPanel, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app     
        self.mainwindow = mainwindow
        self.canvas = canvas
        canvas.switch_text_item.connect(self.on_switch_textitem)
        self.canvas.end_create_textblock.connect(self.onEndCreateTextBlock)
        self.canvas.paste2selected_textitems.connect(self.on_paste2selected_textitems)
        self.canvas.delete_textblks.connect(self.onDeleteBlkItems)
        self.canvas.copy_textblks.connect(self.onCopyBlkItems)
        self.canvas.paste_textblks.connect(self.onPasteBlkItems)
        self.canvas.format_textblks.connect(self.onFormatTextblks)
        self.canvas.layout_textblks.connect(self.onAutoLayoutTextblks)
        self.canvas.reset_angle.connect(self.onResetAngle)
        self.canvas.squeeze_blk.connect(self.onSqueezeBlk)
        self.canvas.path_reorder_finished.connect(
            self.on_path_reorder_finished
        )
        self.canvas.incanvas_selection_changed.connect(
            self._on_canvas_selection_changed
        )
        self.canvas.projective_scale_requested.connect(
            self.on_projective_scale_requested
        )
        self.txtblkShapeControl = canvas.txtblkShapeControl
        self.textpanel = textpanel
        self.textEditList = textpanel.textEditList
        self.textEditList.focus_out.connect(self.on_textedit_list_focusout)
        self.textEditList.textpanel_contextmenu_requested.connect(canvas.on_create_contextmenu)
        self.textEditList.selection_changed.connect(self.on_transwidget_selection_changed)
        self.textEditList.rearrange_blks.connect(self.on_rearrange_blks)
        self.formatpanel = textpanel.formatpanel
        self.formatpanel.textstyle_panel.apply_fontfmt.connect(self.onFormatTextblks)

        self.imgtrans_proj = self.canvas.imgtrans_proj
        self.textblk_item_list: List[TextBlkItem] = []
        self.pairwidget_list: List[TransPairWidget] = self.textEditList.pairwidget_list

        self.auto_textlayout_flag = False
        self.hovering_transwidget : TransTextEdit = None

        self._text_move_snapshot = {}

    def refresh_vertical_layouts(self, _enabled: bool) -> None:
        """Refresh current vertical items after a global layout change."""
        for item in self.textblk_item_list:
            item.refreshVerticalLayout()

    def on_switch_textitem(self, switch_delta: int, key_event: QKeyEvent = None, current_editing_widget: Union[SourceTextEdit, TransTextEdit] = None):
        n_blk = len(self.textblk_item_list)
        if n_blk < 1:
            return
        
        editing_blk = None
        if current_editing_widget is None:
            editing_blk = self.editingTextItem()
            if editing_blk is not None:
                tgt_idx = editing_blk.idx + switch_delta
            else:
                sel_blks = self.canvas.selected_text_items(sort=False)
                if len(sel_blks) == 0:
                    return
                sel_blk = sel_blks[0]
                tgt_idx = sel_blk.idx + switch_delta
        else:
            tgt_idx = current_editing_widget.idx + switch_delta

        if tgt_idx < 0:
            tgt_idx += n_blk
        elif tgt_idx >= n_blk:
            tgt_idx -= n_blk
        blk = self.textblk_item_list[tgt_idx]

        if current_editing_widget is None:
            if editing_blk is None:
                self.canvas.block_selection_signal = True
                self.canvas.clearSelection()
                blk.setSelected(True)
                self.canvas.block_selection_signal = False
                self.canvas.gv.ensureVisible(blk)
                self.txtblkShapeControl.setBlkItem(blk)
                edit = self.pairwidget_list[tgt_idx].e_trans
                self.changeHoveringWidget(edit)
                self.on_incanvas_selection_changed()
            else:
                editing_blk.endEdit()
                editing_blk.setSelected(False)
                self.txtblkShapeControl.setBlkItem(blk)
                blk.setSelected(True)
                blk.startEdit()
                self.canvas.gv.ensureVisible(blk)
        else:
            self.textblk_item_list[current_editing_widget.idx].setSelected(False)
            current_pw = self.pairwidget_list[tgt_idx]
            is_trans = isinstance(current_editing_widget, TransTextEdit)
            if is_trans:
                w = current_pw.e_trans
            else:
                w = current_pw.e_source

            self.changeHoveringWidget(w)
            w.setFocus()

        if key_event is not None:
            key_event.accept()

    def setTextEditMode(self, edit: bool = False) -> None:
        if edit:
            self.textpanel.show()
            self.canvas.textLayer.show()
        else:
            self.canvas.alpha_mask_edit_session.deactivate()
            self.canvas.cancel_path_reorder()
            self.txtblkShapeControl.setBlkItem(None)
            self.textpanel.hide()
            self.textpanel.formatpanel.set_textblk_item()
            self.canvas.textLayer.hide()

    def clearSceneTextitems(
        self,
        reason=SceneTextReplacementReason.CURRENT_PAGE_RELOAD,
    ) -> None:
        self.canvas.text_move_session.cancel()
        self.canvas.alpha_mask_edit_session.deactivate()
        self.canvas.cancel_path_reorder()
        self.canvas.set_primary_selected_text_item(None)
        if reason is not SceneTextReplacementReason.PAGE_CHANGE:
            self.formatpanel.cancel_text_transform_edits_for_scene_change()
        self._text_move_snapshot.clear()
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        for blkitem in self.textblk_item_list:
            blkitem.geometry_controller.release_render_resources()
            self.canvas.removeItem(blkitem)
        self.textblk_item_list.clear()
        self.textEditList.clearAllSelected()
        for textwidget in self.pairwidget_list:
            self.textEditList.removeWidget(textwidget)
        self.pairwidget_list.clear()

    def populateSceneTextitems(self):
        """Create scene items for the project's already-selected current page."""
        self.hovering_transwidget = None
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            self.addTextBlock(textblock)
        if self.auto_textlayout_flag:
            self.updateTextBlkList()

    def updateSceneTextitems(
        self,
        reason=SceneTextReplacementReason.CURRENT_PAGE_RELOAD,
    ):
        """Replace all scene text items at an explicit lifecycle boundary."""
        if reason is SceneTextReplacementReason.CURRENT_PAGE_RELOAD:
            # Commands retain their original QGraphicsItems. They cannot remain
            # valid after the current page is rebuilt with new item instances.
            self.canvas.clear_text_stack()
        elif reason is SceneTextReplacementReason.PROJECT_RELOAD:
            self.canvas.clear_undostack(update_saved_step=True)
        self.clearSceneTextitems(reason)
        self.populateSceneTextitems()

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            translation = ''
            if self.auto_textlayout_flag and not blk.vertical:
                translation = blk.translation
                blk.translation = ''
                # Layout must fit against the block's detected box; persisted
                # rich_text would size the item to the previous render instead.
                blk.rich_text = ''
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
            if translation:
                blk.translation = translation
                rst = self.layout_textblk(blk_item, text=translation)
                if rst is None:
                    blk_item.setPlainText(translation)
        self.addTextBlkItem(blk_item)

        pair_widget = TransPairWidget(len(self.pairwidget_list), pcfg.fold_textarea)
        self.pairwidget_list.append(pair_widget)
        self.textEditList.addPairWidget(pair_widget)
        pair_widget.e_source.setPlainText(blk_item.blk.get_text())
        pair_widget.e_source.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_source.focus_out.connect(self.on_pairw_focusout)

        pair_widget.e_trans.setPlainText(blk_item.toPlainText())
        pair_widget.e_trans.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_trans.propagate_user_edited.connect(self.on_propagate_transwidget_edit)
        pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_trans.focus_out.connect(self.on_pairw_focusout)
        pair_widget.drag_move.connect(self.textEditList.handle_drag_pos)
        pair_widget.pw_drop.connect(self.textEditList.on_pw_dropped)
        pair_widget.idx_edited.connect(self.textEditList.on_idx_edited)

        self.new_textblk.emit(blk_item.idx)
        return blk_item

    def addTextBlkItem(self, textblk_item: TextBlkItem) -> TextBlkItem:
        self.textblk_item_list.append(textblk_item)
        self.canvas.attach_text_item(textblk_item)
        textblk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
        textblk_item.end_edit.connect(self.onTextBlkItemEndEdit)
        textblk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
        textblk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
        textblk_item.move_interaction_finished.connect(
            self.onTextBlkItemMoveFinished
        )
        textblk_item.reshaped.connect(self.onTextBlkItemReshaped)
        textblk_item.rotated.connect(self.onTextBlkItemRotated)
        textblk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
        textblk_item.undo_signal.connect(self.on_textedit_undo)
        textblk_item.redo_signal.connect(self.on_textedit_redo)
        textblk_item.propagate_user_edited.connect(self.on_propagate_textitem_edit)
        textblk_item.inline_format_changed.connect(
            self.on_inline_format_changed
        )
        textblk_item.pasted.connect(self.onBlkitemPaste)
        return textblk_item

    def deleteTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        selection_changed = False
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            if blkitem.isSelected():
                selection_changed = True
            self.canvas.removeItem(blkitem) # removeItem itself will block incanvas_selection_changed
            self.textblk_item_list.remove(blkitem)
            self.pairwidget_list.remove(p_widget)
            self.textEditList.removeWidget(p_widget)
        self.updateTextBlkItemIdx()
        self.txtblkShapeControl.setBlkItem(None)
        if selection_changed:
            # it must be called after updateTextBlkItemIdx if blk.idx changed
            self.on_incanvas_selection_changed()

    def recoverTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        self.canvas.block_selection_signal = True
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.textblk_item_list.insert(blkitem.idx, blkitem)
            self.canvas.attach_text_item(blkitem)
            self.pairwidget_list.insert(p_widget.idx, p_widget)
            self.textEditList.insertPairWidget(p_widget, p_widget.idx)
            if self.txtblkShapeControl.blk_item is not None and blkitem.isSelected():
                blkitem.setSelected(False)
        self.updateTextBlkItemIdx()
        self.on_incanvas_selection_changed()
        self.canvas.block_selection_signal = False
        
    @property
    def app_clipborad(self) -> QClipboard:
        return self.app.clipboard()

    def onBlkitemPaste(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        if blk_item.insert_from_mime_data(self.app_clipborad.mimeData()):
            return
        text = self.app_clipborad.text()
        blk_item.insert_plain_text_at_cursor(text)

    def on_inline_format_changed(self) -> None:
        item = self.sender()
        if item is self.formatpanel.textblk_item:
            self.formatpanel.sync_inline_format(item.get_fontformat())

    def onTextBlkItemBeginEdit(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.formatpanel.text_transform_session.select_transform(-1)
        self.txtblkShapeControl.setBlkItem(blk_item)
        self.canvas.editing_textblkitem = blk_item
        self.formatpanel.set_textblk_item(blk_item)
        self.txtblkShapeControl.startEditing()
        e_trans = self.pairwidget_list[blk_item.idx].e_trans
        self.changeHoveringWidget(e_trans)

    def changeHoveringWidget(self, edit: SourceTextEdit):
        if self.hovering_transwidget is not None and self.hovering_transwidget != edit:
            self.hovering_transwidget.setHoverEffect(False)
        self.hovering_transwidget = edit
        if edit is not None:
            pw = self.pairwidget_list[edit.idx]
            h = pw.height()
            if shared.USE_PYSIDE6:
                self.textEditList.ensureWidgetVisible(pw, ymargin=h)
            else:
                self.textEditList.ensureWidgetVisible(pw, yMargin=h)
            edit.setHoverEffect(True)

    def onLeftbuttonPressed(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.canvas.set_primary_selected_text_item(blk_item)
        self.txtblkShapeControl.setBlkItem(blk_item)
        selections = self.canvas.selected_text_items(sort=False)
        if blk_item not in selections:
            selections.append(blk_item)
        self._text_move_snapshot = {
            item: QPointF(item.logical_position()) for item in selections
        }
        for item in selections:
            item._old_pos = item.pos()
        self.changeHoveringWidget(self.pairwidget_list[blk_id].e_trans)

    def onTextBlkItemEndEdit(self, blk_id: int):
        self.canvas.editing_textblkitem = None
        self.textblk_item_list[blk_id].setSelected(True)
        self.txtblkShapeControl.endEditing()

    def editingTextItem(self) -> TextBlkItem:
        if self.txtblkShapeControl.isVisible() and self.canvas.editing_textblkitem is not None:
            return self.canvas.editing_textblkitem
        return None

    def is_editting(self) -> bool:
        blk_item = self.txtblkShapeControl.blk_item
        return blk_item is not None and blk_item.isEditing()

    def onTextBlkItemHoverEnter(self, blk_id: int):
        if self.is_editting():
            return
        blk_item = self.textblk_item_list[blk_id]
        if self.canvas.active_transform_control_item() is blk_item:
            shape = self.txtblkShapeControl
            if shape.blk_item is not blk_item:
                shape.setBlkItem(blk_item)
            shape.hide()
            return
        if not blk_item.hasFocus():
            self.txtblkShapeControl.setBlkItem(blk_item)

    def onTextBlkItemMoveFinished(self):
        try:
            items = [
                item
                for item in self._text_move_snapshot
                if item.scene() is self.canvas
            ]
            if not items:
                return
            before = [
                QPointF(
                    self._text_move_snapshot.get(
                        item, item.logical_position()
                    )
                )
                for item in items
            ]
            after = [QPointF(item.logical_position()) for item in items]
            if before != after:
                self.canvas.push_undo_command(
                    MoveBlkItemsCommand(
                        items,
                        before_positions=before,
                        after_positions=after,
                    )
                )
        finally:
            self._text_move_snapshot.clear()
        
    def onTextBlkItemReshaped(self, item: TextBlkItem):
        self.canvas.push_undo_command(
            ReshapeItemCommand(item)
        )

    def onTextBlkItemRotated(self, new_angle: float):
        blk_item = self.txtblkShapeControl.blk_item
        if blk_item:
            self.canvas.push_undo_command(
                RotateItemCommand(
                    blk_item,
                    new_angle,
                )
            )

    def onDeleteBlkItems(self, mode: int) -> None:
        # Recovery masks and undo rectangles must use committed geometry;
        # command construction can capture or mutate items before stack.push().
        self.formatpanel.resolve_text_transform_edits_for_history_change()
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(DeleteBlkItemsCommand(selected_blks, mode, self))

    def onCopyBlkItems(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)

        if len(selected_blks) == 0:            
            return

        self.canvas.clipboard_blks.clear()
        if self.canvas.text_change_unsaved():
            self.updateTextBlkList()

        pos = selected_blks[0].blk.bounding_rect()
        pos_x = int(pos[0] + pos[2] / 2)
        pos_y = int(pos[1] + pos[3] / 2)

        textlist = []
        for blkitem in selected_blks:
            blk = copy.deepcopy(blkitem.blk)
            blk.adjust_pos(-pos_x, -pos_y)
            self.canvas.clipboard_blks.append(blk)
            textlist.append(blkitem.toPlainText().strip())
        textlist = '\n'.join(textlist)
        self.app_clipborad.setText(textlist, QClipboard.Mode.Clipboard)


    def onPasteBlkItems(self, pos: QPointF):
        if pos is None:
            pos_x, pos_y = 0, 0
        else:
            pos_x, pos_y = pos.x(), pos.y()
            pos_x = int(pos_x / self.canvas.scale_factor)
            pos_y = int(pos_y / self.canvas.scale_factor)
        blkitem_list, pair_widget_list = [], []
        for blk in self.canvas.clipboard_blks:
            blk = copy.deepcopy(blk)
            blk.adjust_pos(pos_x, pos_y)
            blkitem = self.addTextBlock(blk)
            pairw = self.pairwidget_list[-1]
            blkitem_list.append(blkitem)
            pair_widget_list.append(pairw)
        if len(blkitem_list) > 0:
            self.canvas.clearSelection()
            self.canvas.push_undo_command(PasteBlkItemsCommand(blkitem_list, pair_widget_list, self))
            if len(blkitem_list) == 1:
                self.formatpanel.set_textblk_item(blkitem_list[0])
            else:
                self.formatpanel.set_textblk_item(multi_select=True)

    def onFormatTextblks(self, fmt: FontFormat = None):
        if fmt is None:
            fmt = self.formatpanel.global_format
        self.apply_fontformat(fmt)

    def onAutoLayoutTextblks(self) -> None:
        self.formatpanel.resolve_text_transform_edits_for_history_change()
        selected_blks = self.canvas.selected_text_items()
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if len(selected_blks) > 0:
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                self.layout_textblk(blkitem)

            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def onResetAngle(self) -> None:
        self.formatpanel.resolve_text_transform_edits_for_history_change()
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(
                ResetAngleCommand(selected_blks)
            )

    def onSqueezeBlk(self) -> None:
        self.formatpanel.resolve_text_transform_edits_for_history_change()
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(SqueezeCommand(selected_blks, self.txtblkShapeControl))

    def _on_canvas_selection_changed(self):
        self.on_incanvas_selection_changed()

    def on_incanvas_selection_changed(self) -> None:
        if self.canvas.textEditMode():
            textitems = self.canvas.selected_text_items()
            self.textEditList.set_selected_list([t.idx for t in textitems])
            self._update_selection_panels(textitems)

    def _update_selection_panels(
        self,
        textitems: List[TextBlkItem],
        primary_item: Optional[TextBlkItem] = None,
    ) -> None:
        if len(textitems) == 1:
            self.formatpanel.set_textblk_item(textitems[-1])
        else:
            if primary_item is None:
                primary_item = self.canvas.primary_selected_text_item(textitems)
            self.formatpanel.set_textblk_item(
                multi_select=bool(textitems), primary_item=primary_item
            )

    def on_projective_scale_requested(self, item: TextBlkItem) -> None:
        session = self.formatpanel.text_transform_session
        if len(session.items) != 1 or session.items[0] is not item:
            self._update_selection_panels([item])
        session.activate_last_projective(item)

    def layout_textblk(self, blkitem: TextBlkItem, text: str = None, mask: np.ndarray = None, bounding_rect: List = None, region_rect: List = None):
        
        '''
        auto text layout, vertical writing is not supported yet.
        '''

        img = self.imgtrans_proj.img_array
        if img is None:
            return
        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)

        # disable for vertical writing
        if blkitem.blk.vertical:
            return
        
        old_br = blkitem.absBoundingRect(qrect=True)
        old_br = [old_br.x(), old_br.y(), old_br.width(), old_br.height()]
        if old_br[2] < 1:
            return

        blk_font = blkitem.font()
        orig_font_size = blk_font.pointSizeF()
        fmt = blkitem.get_fontformat()
        blk_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, fmt.letter_spacing * 100)
        text_size_func = lambda text: get_text_size(QFontMetricsF(blk_font), text)

        restore_charfmts = False
        if text is None:
            text = blkitem.toPlainText()
            restore_charfmts = True

        if not text.strip():
            return

        bubble = None
        if mask is None:
            im_h, im_w = img.shape[:2]
            # Fit against the detection box, not the item rect: squeeze and
            # re-layout shrink the item toward its text, and a window built
            # from it ratchets down until growth has no room left. The
            # detection box stays put until the next detect run. Rich-text
            # init and oversized blocks can push it past the page edge, and
            # enlarge_window inverts when that happens - clamp first.
            detected_box = getattr(blkitem.blk, '_detected_bbox', None)
            bounding_rect = list(detected_box) if detected_box else list(blkitem.blk.bounding_rect())
            bx, by = max(bounding_rect[0], 0), max(bounding_rect[1], 0)
            bw = min(bounding_rect[0] + bounding_rect[2], im_w) - bx
            bh = min(bounding_rect[1] + bounding_rect[3], im_h) - by
            if bw > 0 and bh > 0:
                bounding_rect = [bx, by, bw, bh]
            else:
                bounding_rect = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
            if bounding_rect[2] <= 0 or bounding_rect[3] <= 0:
                blkitem.setPlainText(text)
                if len(self.pairwidget_list) > blkitem.idx:
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(text)
                return
            if tgt_is_cjk:
                max_enlarge_ratio = 2.5
            else:
                max_enlarge_ratio = 3
            enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, max_enlarge_ratio)
            mask, ballon_area, mask_xyxy, region_rect = extract_ballon_region(img, bounding_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True)
            bubble = _bubble_polygon_for(self.imgtrans_proj, blkitem.blk)
            if bubble is not None:
                # Detector geometry wins over the flood-fill: no window
                # clipping, no glyph holes, exact centroid for centering.
                poly_mask = np.zeros(img.shape[:2], np.uint8)
                cv2.fillPoly(poly_mask, [np.asarray(bubble, np.int32)], 255)
                px1, py1, px2, py2 = mask_xyxy
                crop = poly_mask[py1:py2, px1:px2]
                if crop.any():
                    # One block owns this outline: fit against the whole
                    # bubble. The detection-box window clips balloons larger
                    # than the box, and the clipped bbox caps growth early.
                    # A polygon holding several blocks is a joined
                    # multi-bubble; its window band keeps each block near
                    # its own text instead of re-centering both on the seam.
                    # Holders = blocks whose deepest containment IS this
                    # polygon: merely touching an overlapping neighbour must
                    # not demote a single-owner outline to the clipped band.
                    page_blocks = getattr(self.imgtrans_proj, 'pages', None) or {}
                    page_blocks = page_blocks.get(self.imgtrans_proj.current_img, [])
                    holders = sum(
                        1 for other in page_blocks
                        if _bubble_polygon_for(self.imgtrans_proj, other) == bubble
                    )
                    if holders <= 1:
                        bx, by, bw, bh = cv2.boundingRect(np.asarray(bubble, np.int32))
                        mask_xyxy = [bx, by, bx + bw, by + bh]
                        mask = poly_mask[by:by + bh, bx:bx + bw]
                        ballon_area = int(mask.nonzero()[0].size)
                    else:
                        mask = crop
                        ballon_area = int(crop.nonzero()[0].size)
        else:
            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]
        poly_arr = np.asarray(bubble, np.float32) if bubble is not None else None
        # One nonzero pass feeds both the hyphen line budget and the fit loop.
        mask_ys, mask_xs = np.nonzero(mask)
        mb_x0 = mb_y0 = mb_x1 = mb_y1 = mb_w = mb_h = 0
        if mask_ys.size:
            mb_x0, mb_x1 = int(mask_xs.min()), int(mask_xs.max()) + 1
            mb_y0, mb_y1 = int(mask_ys.min()), int(mask_ys.max()) + 1
            mb_w, mb_h = mb_x1 - mb_x0, mb_y1 - mb_y0
        
        words, delimiter = seg_text(text, pcfg.module.translate_target)
        if len(words) < 1:
            return

        wl_list = get_words_length_list(QFontMetricsF(blk_font), words)
        text_w, text_h = text_size_func(text)
        text_area = text_w * text_h
        if tgt_is_cjk:
            line_height = int(round(fmt.line_spacing * text_size_func('X木')[1]))
        else:
            line_height = int(round(fmt.line_spacing * text_size_func('X')[1]))
        delimiter_len = text_size_func(delimiter)[0]
 
        ref_src_lines = False
        if not blkitem.blk.src_is_vertical:
            ref_src_lines = blkitem.blk.line_coord_valid(old_br)

        adaptive_fntsize = False
        resize_ratio = 1
        if self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag:
            if blkitem.blk.src_is_vertical and blkitem.blk.vertical != blkitem.blk.src_is_vertical:
                adaptive_fntsize = True
                area_ratio = ballon_area / text_area
                ballon_area_thresh = 1.7
                downscale_constraint = 0.6
                resize_ratio = np.clip(min(area_ratio / ballon_area_thresh, region_rect [2] / max(wl_list)), downscale_constraint, 1.0)

            else:
                if not src_is_cjk:
                    resize_ratio_ballon = max(ballon_area / 1.2 / text_area, 0.7)
                    if ref_src_lines:
                        _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                        resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                        resize_ratio = min(resize_ratio_ballon, resize_ratio_src)
                    else:
                        resize_ratio = resize_ratio_ballon
                elif not blkitem.blk.src_is_vertical and ref_src_lines:
                    _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                    resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                    resize_ratio = max(resize_ratio_src * 1.5, 0.5)
                resize_ratio = min(max(resize_ratio, 0.6), 1)

        if resize_ratio != 1:
            new_font_size = blk_font.pointSizeF() * resize_ratio
            blk_font.setPointSizeF(new_font_size)
            wl_list = (np.array(wl_list, np.float64) * resize_ratio).astype(np.int32).tolist()
            line_height = int(line_height * resize_ratio)
            text_w = int(text_w * resize_ratio)
            delimiter_len = int(delimiter_len * resize_ratio)

        # Optical line-end hyphenation for Latin scripts: the wrap loops
        # close a line with a word's hyphen-fitting head when the whole
        # word would break the budget. CJK scripts never hyphenate.
        hyphenator = None
        hyphen_measure = None
        if not tgt_is_cjk:
            hyphenator = hyphenator_for_target(pcfg.module.translate_target)
            if hyphenator is not None:
                hyphen_measure = lambda s: text_size_func(s)[0]

        # Latin-script hyphenation: after the font settles, split tokens that
        # still exceed the line budget at linguistic points. CJK scripts wrap
        # per character and never hyphenate.
        if not tgt_is_cjk and words:
            budget = mb_w if mb_w > 0 else mask.shape[1]
            words, wl_list = hyphenate_long_words(
                words, wl_list, lambda s: text_size_func(s)[0],
                pcfg.module.translate_target, budget
            )

        max_central_width = np.inf
        if fmt.alignment == 1:
            if len(blkitem.blk) > 0:
                centroid = blkitem.blk.center().astype(np.int64).tolist()
                centroid[0] -= mask_xyxy[0]
                centroid[1] -= mask_xyxy[1]
            else:
                centroid = [bounding_rect[2] // 2, bounding_rect[3] // 2]
            if poly_arr is not None and mask_ys.size:
                # Center on the bubble's bounding-box middle: the pixel mean
                # drifts toward lopsided mass and the deepest point chases
                # the thickest stroke; the bbox middle is what reads as
                # "centered" in a balloon (mask coords are window-relative).
                centroid = [(mb_x0 + mb_x1) // 2, (mb_y0 + mb_y1) // 2]
        else:
            max_central_width = np.inf
            centroid = [0, 0]
            abs_centroid = [bounding_rect[0], bounding_rect[1]]
            if len(blkitem.blk) > 0:
                blkitem.blk.lines[0]
                abs_centroid = blkitem.blk.lines[0][0]
                centroid[0] = int(abs_centroid[0] - mask_xyxy[0])
                centroid[1] = int(abs_centroid[1] - mask_xyxy[1])

        row_profile = None
        if (poly_arr is not None and mb_w > 0
                and abs(blkitem.blk.angle) == 0
                and pcfg.let_elliptic_layout):
            # Exact outline width per row, measured once: each line is
            # budgeted by the bubble's real extent at its own height, so
            # round, rectangular, lopsided, and concave balloons all fit
            # without a fitted-ellipse fudge factor. The flag still turns
            # shape-aware budgeting off entirely, falling back to the mask
            # rectangle.
            row_profile = row_width_profile(poly_arr, mb_y0, mb_y1 + 1)
            # Wrap column: the widest row the outline offers. A single
            # long line would otherwise use full width while the height
            # stays empty, and growth becomes width-bound.
            max_central_width = float(np.nanmax(
                np.where(row_profile[1] > row_profile[0],
                         row_profile[1] - row_profile[0], 0.0)
            ))

        # Layout, then keep shrinking until the final canvas fits the balloon
        # mask: the ratio heuristic above is only a guess (its 0.6/0.7 floors
        # alone leave long translations overflowing), and layout_text stacks
        # lines with no vertical limit. Fresh word lists per attempt, because
        # layout_lines pops from them. Rotated blocks lay out once and keep the
        # legacy width-only adjustment below: layout_text wraps in a rotated
        # frame this mask check cannot sample.
        check_fit = (
            self.auto_textlayout_flag
            and pcfg.let_fntsize_flag == 0
            and pcfg.let_autolayout_flag
            and abs(blkitem.blk.angle) == 0
        )
        if check_fit and mask_ys.size == 0:
            check_fit = False

        offset_retries = 0
        prev_frac = -1.0
        grew = False
        good_text, good_xywh, good_font, good_ratio = None, None, 0.0, 1.0
        # Coverage is measured on a hole-closed mask: the source image still
        # carries the original glyphs the flood fill leaves out, and counting
        # them as off-mask makes an in-balloon text look under-covered.
        k = max(5, int(np.sqrt(max(int((mask > 0).sum()), 1)) / 12))
        measure_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) if check_fit else mask
        for _ in range(13 if check_fit else 1):
            new_text, xywh, start_from_top, adjust_xy = layout_text(
                blkitem.blk,
                mask,
                mask_xyxy,
                centroid,
                list(words),
                list(wl_list),
                delimiter,
                delimiter_len,
                line_height,
                0,
                max_central_width,
                src_is_cjk=src_is_cjk,
                tgt_is_cjk=tgt_is_cjk,
                ref_src_lines=ref_src_lines,
                row_profile=row_profile,
                hyphenator=hyphenator,
                measure=hyphen_measure
            )
            if not check_fit:
                break
            x, y, w, h = xywh
            lx, ly = x - mask_xyxy[0], y - mask_xyxy[1]
            inside = 0 <= lx and 0 <= ly and lx + w <= mask.shape[1] and ly + h <= mask.shape[0]
            frac = float((measure_mask[ly:ly + h, lx:lx + w] > 0).mean()) if inside and h > 0 and w > 0 else -1.0
            # Fit target: the canvas must be inside the window with most of
            # its area on the mask. Shrink until coverage reaches the target
            # or stops improving (glyph holes in the source image, irregular
            # balloon shapes) - a fixed bbox-plus-partial-coverage rule let
            # up to 40% of the text hang outside round balloons.
            if poly_arr is not None:
                # Collision with the detector outline, per rendered line: the
                # first word of a line lands before any width cap runs, and a
                # bounding box around curved text always has corners outside
                # the outline - so probe each line's own rect (corners plus
                # edge midpoints; non-convex dents slip between corners).
                line_texts = new_text.split('\n')
                line_wl = [text_size_func(t)[0] for t in line_texts]
                if fmt.alignment == 1:
                    line_rects = [
                        (x + (w - lw) // 2, y + i * line_height, lw, line_height)
                        for i, lw in enumerate(line_wl)
                    ]
                elif fmt.alignment == 2:
                    line_rects = [
                        (x + w - lw, y + i * line_height, lw, line_height)
                        for i, lw in enumerate(line_wl)
                    ]
                else:
                    line_rects = [
                        (x, y + i * line_height, lw, line_height)
                        for i, lw in enumerate(line_wl)
                    ]
                clear = True
                for rx, ry, rw, rh in line_rects:
                    # One-pixel inset: the rendered line box carries a pixel
                    # of padding the layout rows do not, so a corner landing
                    # exactly on the outline would spill in the render.
                    ix0, iy0 = rx + 1, ry + 1
                    ix1, iy1 = rx + max(rw - 1, 1), ry + max(rh - 1, 1)
                    for qx, qy in (
                        (ix0, iy0), (ix1, iy0), (ix0, iy1), (ix1, iy1),
                        ((ix0 + ix1) // 2, iy0), ((ix0 + ix1) // 2, iy1),
                        (ix0, (iy0 + iy1) // 2), (ix1, (iy0 + iy1) // 2),
                    ):
                        if cv2.pointPolygonTest(poly_arr, (float(qx), float(qy)), False) < 0:
                            clear = False
                            break
                    if not clear:
                        break
                if inside and clear:
                    # Accepted: keep growing toward the outline when the
                    # starting font leaves the bubble mostly empty (the
                    # pre-fit heuristic only ever shrinks). Probes police
                    # every grown step and a failed look-ahead reverts below;
                    # no growth on the last two iterations, so the final
                    # layout always matches the applied font.
                    room = min(mb_w * 0.98 / w, mb_h * 0.98 / h)
                    if room > 1.03 and _ < 11:
                        good_text, good_xywh = new_text, xywh
                        good_font = blk_font.pointSizeF()
                        good_ratio = resize_ratio
                        grow = min(room, 1.15)
                        resize_ratio *= grow
                        blk_font.setPointSizeF(good_font * grow)
                        wl_list = (np.array(wl_list, np.float64) * grow).astype(np.int32).tolist()
                        line_height = int(line_height * grow)
                        delimiter_len = int(delimiter_len * grow)
                        grew = True
                        continue
                    break
            elif inside and frac >= 0.9:
                break
            if poly_arr is None and inside and prev_frac >= 0 and frac - prev_frac < 0.015:
                # Coverage plateau: the mask will not take more text. Outline
                # fits terminate on the collision probes or the readability
                # floor instead - a plateau here would accept exactly the
                # lines the probes just rejected (a single word wider than
                # the bubble at the fitted size).
                break
            prev_frac = frac if inside else -1.0
            if grew:
                # The grown look-ahead failed the collision test: restore the
                # last accepted layout. Shrinking from the overshoot would
                # oscillate around the fill target.
                new_text, xywh = good_text, good_xywh
                blk_font.setPointSizeF(good_font)
                resize_ratio = good_ratio
                break
            # Scale toward the mask bbox; when the bbox size already fits,
            # only corner spill and placement offset remain. Corner spill
            # shrinks away; an anchored origin that stays off the mask after
            # one more shrink never will - accept and stop (free-standing
            # credits text can sit offset from whatever white region floods
            # behind it, and chasing that shrinks them to the floor).
            # A degenerate layout (zero canvas at absurd shrink sizes)
            # must not divide by zero; treat it as needing a shrink step.
            scale = min(mb_w * 0.98 / w, mb_h * 0.98 / h) if w > 0 and h > 0 else 0.9
            if scale >= 1:
                off_mask = lx < mb_x0 or ly < mb_y0 or lx + w > mb_x1 or ly + h > mb_y1
                if off_mask:
                    offset_retries += 1
                    # Centered text re-converges on the balloon as it shrinks;
                    # only side-anchored placement can be stuck off the mask.
                    if offset_retries > 1 and fmt.alignment != 1:
                        break
                scale = 0.9
            # Clamp to the readability floor instead of skipping the shrink:
            # one coarse ratio step can overshoot the floor even when the
            # floor itself still fits the balloon (big fresh-detected source
            # fonts need a large first step). 0.15 keeps text recognizable
            # while letting canvas converge onto boxes whose center sits
            # off the balloon center.
            min_size = orig_font_size * 0.15
            if blk_font.pointSizeF() * scale < min_size:
                scale = min_size / blk_font.pointSizeF()
            if scale >= 1:
                break  # already at the floor
            resize_ratio *= scale
            blk_font.setPointSizeF(blk_font.pointSizeF() * scale)
            wl_list = (np.array(wl_list, np.float64) * scale).astype(np.int32).tolist()
            # Keep at least one pixel: int truncation can zero the row pitch
            # after enough shrink steps, which breaks every later layout.
            line_height = max(1, int(line_height * scale))
            delimiter_len = max(1, int(delimiter_len * scale))

        # font size post adjustment
        post_resize_ratio = 1
        if adaptive_fntsize and not check_fit:
            downscale_constraint = 0.5
            w = xywh[2]
            post_resize_ratio = np.clip(max(region_rect[2] / w, downscale_constraint), 0, 1)
            resize_ratio *= post_resize_ratio

        if post_resize_ratio != 1:
            cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
            w, h = xywh[2] * post_resize_ratio, xywh[3] * post_resize_ratio
            xywh = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

        if resize_ratio != 1:
            new_font_size = blkitem.font().pointSizeF() * resize_ratio
            blkitem.textCursor().clearSelection()
            blkitem.setFontSize(new_font_size)
            blk_font.setPointSizeF(new_font_size)

        if restore_charfmts:
            char_fmts = blkitem.get_char_fmts()        
        
        ffmt = QFontMetricsF(blk_font)
        maxw = max([ffmt.horizontalAdvance(t) for t in new_text.split('\n')])
        blkitem.set_size(maxw * 1.5, xywh[3], set_layout_maxsize=True)
        blkitem.setPlainText(new_text)
        if len(self.pairwidget_list) > blkitem.idx:
            self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
        if restore_charfmts:
            self.restore_charfmts(blkitem, text, new_text, char_fmts)
        if resize_ratio != 1:
            # setPlainText rebuilds the document; when it held persisted
            # rich_text the document default font stays stale, so re-apply
            # the fitted size to the new text and re-sync the default (font()
            # reads it, and the next layout pass bases on it).
            blkitem.setFontSize(blk_font.pointSizeF())
            blkitem.document().setDefaultFont(blk_font)
        blkitem.squeezeBoundingRect()
        # Center the settled text box on the balloon itself: the layout
        # anchors on the detection box center, which is off the balloon
        # center for most bubbles, and the item rect (not layout_text's xywh)
        # is what the canvas renders. Re-measure after each move: the first
        # moveBy re-anchors the rect through the geometry controller.
        for _ in range(3):
            _ys, _xs = np.nonzero(mask)
            if _ys.size == 0:
                break
            _cx = mask_xyxy[0] + (mb_x0 + mb_x1) / 2
            _cy = mask_xyxy[1] + (mb_y0 + mb_y1) / 2
            _br = blkitem.absBoundingRect(qrect=True)
            _dx = _cx - (_br.x() + _br.width() / 2)
            _dy = _cy - (_br.y() + _br.height() / 2)
            if abs(_dx) <= 1 and abs(_dy) <= 1:
                break
            blkitem.moveBy(_dx, _dy)
        return True
    
    def restore_charfmts(self, blkitem: TextBlkItem, text: str, new_text: str, char_fmts: List[QTextCharFormat]):
        cursor = blkitem.textCursor()
        cpos = 0
        num_text = len(new_text)
        num_fmt = len(char_fmts)
        blkitem.layout.relayout_on_changed = False
        blkitem.repaint_on_changed = False
        if num_text >= num_fmt:
            for fmt_i in range(num_fmt):
                fmt = char_fmts[fmt_i]
                ori_char = text[fmt_i].strip()
                if ori_char == '':
                    continue
                else:
                    if cursor.atEnd():   
                        break
                    matched = False
                    while cpos < num_text:
                        if new_text[cpos] == ori_char:
                            matched = True
                            break
                        cpos += 1
                    if matched:
                        cursor.clearSelection()
                        cursor.setPosition(cpos)
                        cursor.setPosition(cpos+1, QTextCursor.MoveMode.KeepAnchor)
                        cursor.setCharFormat(fmt)
                        cursor.setBlockCharFormat(fmt)
                        cpos += 1
        blkitem.repaint_on_changed = True
        blkitem.layout.relayout_on_changed = True
        blkitem.layout.reLayout()
        blkitem.repaint_background()

    def onEndCreateTextBlock(self, rect: QRectF):
        xyxy = np.array([rect.x(), rect.y(), rect.right(), rect.bottom()])        
        xyxy = np.round(xyxy).astype(np.int32)
        block = TextBlock(xyxy)
        xywh = np.copy(xyxy)
        xywh[[2, 3]] -= xywh[[0, 1]]
        block.set_lines_by_xywh(xywh)
        block.src_is_vertical = self.formatpanel.global_format.vertical
        blk_item = TextBlkItem(block, len(self.textblk_item_list), set_format=False, show_rect=True)
        blk_item.set_fontformat(self.formatpanel.global_format)
        self.canvas.push_undo_command(CreateItemCommand(blk_item, self))

    def on_paste2selected_textitems(self):
        blkitems = self.canvas.selected_text_items()
        text = self.app_clipborad.text()

        num_blk = len(blkitems)
        if num_blk < 1:
            return
        
        if num_blk > 1:
            text_list = text.rstrip().split('\n')
            num_text = len(text_list)
            if num_text > 1:
                if num_text > num_blk:
                    text_list = text_list[:num_blk]
                elif num_text < num_blk:
                    text_list = text_list + [text_list[-1]] * (num_blk - num_text)
                text = text_list
        
        etrans = [self.pairwidget_list[blkitem.idx].e_trans for blkitem in blkitems]
        self.canvas.push_undo_command(MultiPasteCommand(text, blkitems, etrans))

    def capitalize_selected_textitems(self) -> None:
        """Capitalize selected translations as one synchronized undo action."""
        if not self.canvas.textEditMode():
            return
        items = self.canvas.selected_text_items()
        edits = [self.pairwidget_list[item.idx].e_trans for item in items]
        command = CapitalizeTextItemsCommand.create(items, edits)
        if command is not None:
            self.canvas.push_undo_command(command)

    def on_transwidget_focus_in(self, idx: int) -> None:
        self.canvas.cancel_path_reorder()
        if self.is_editting():
            textitm = self.editingTextItem()
            textitm.endEdit()
            self.pairwidget_list[textitm.idx].e_trans.setHoverEffect(False)
            self.textEditList.clearAllSelected()

        if idx < len(self.textblk_item_list):
            blk_item = self.textblk_item_list[idx]
            self.canvas.gv.ensureVisible(blk_item)
            self.txtblkShapeControl.setBlkItem(blk_item)

    def on_textedit_redo(self):
        self.canvas.redo_textedit()

    def on_textedit_undo(self):
        self.canvas.undo_textedit()

    def on_pairw_focusout(self, idx: int):
        sender = self.sender()
        if isinstance(sender, TransTextEdit) and idx < len(self.textblk_item_list):
            blk_item = self.textblk_item_list[idx]
            blk_item.refresh_cache_policy()

    def on_push_textitem_undostack(self, num_steps: int, is_formatting: bool):
        blkitem: TextBlkItem = self.sender()
        e_trans = self.pairwidget_list[blkitem.idx].e_trans if not is_formatting else None
        self.canvas.push_undo_command(TextItemEditCommand(blkitem, e_trans, num_steps, self.textpanel.formatpanel), update_pushed_step=is_formatting)

    def on_push_edit_stack(self, num_steps: int):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        is_trans = type(edit) == TransTextEdit
        blkitem = self.textblk_item_list[edit.idx] if is_trans else None
        self.canvas.push_undo_command(TextEditCommand(edit, num_steps, blkitem), update_pushed_step=not is_trans)

    def on_propagate_textitem_edit(
        self,
        pos: int,
        removed: int,
        added_text: str,
        joint_previous: bool,
    ) -> None:
        blk_item: TextBlkItem = self.sender()
        edit = self.pairwidget_list[blk_item.idx].e_trans
        propagate_user_edit(
            edit, pos, removed, added_text, joint_previous
        )
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def on_propagate_transwidget_edit(
        self,
        pos: int,
        removed: int,
        added_text: str,
        joint_previous: bool,
    ) -> None:
        edit: TransTextEdit = self.sender()
        blk_item = self.textblk_item_list[edit.idx]
        if blk_item.isEditing():
            blk_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        propagate_user_edit(
            blk_item, pos, removed, added_text, joint_previous
        )
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def apply_fontformat(self, fontformat: FontFormat) -> None:
        """Apply one whole format after settling transient edit owners.

        >>> callable(SceneTextManager.apply_fontformat)
        True
        """
        selected_blks = self.canvas.selected_text_items()
        trans_widget_list = []
        for blk in selected_blks:
            trans_widget_list.append(self.pairwidget_list[blk.idx].e_trans)
        if len(selected_blks) > 0:
            # Whole-format replacement can reindex or remove Image effects;
            # settle the same transient owners used by undo/redo first.
            self.formatpanel.resolve_text_transform_edits_for_history_change()
            self.canvas.push_undo_command(
                ApplyFontformatCommand(
                    selected_blks,
                    trans_widget_list,
                    fontformat,
                )
            )
            if self.formatpanel.global_mode():
                if id(self.formatpanel.active_text_style_format()) != id(fontformat):
                    self.formatpanel.deactivate_style_label()
                self.formatpanel.on_active_textstyle_label_changed()
            else:
                self.formatpanel.set_active_format(fontformat)

    def on_transwidget_selection_changed(self) -> None:
        editing_item = self.canvas.editing_textblkitem
        if editing_item is not None and editing_item.isEditing():
            # end_edit reselects its item, so finish it before applying the
            # pair list's authoritative selection.
            editing_item.endEdit()

        selitems = self.canvas.selected_text_items()
        selset = {pw.idx: pw for pw in self.textEditList.checked_list}
        self.canvas.block_selection_signal = True
        try:
            for blkitem in selitems:
                if blkitem.idx not in selset:
                    blkitem.setSelected(False)
                else:
                    selset.pop(blkitem.idx)
            for idx in selset:
                self.textblk_item_list[idx].setSelected(True)
        finally:
            self.canvas.block_selection_signal = False
        selected = self.canvas.selected_text_items()
        anchor = self.textEditList.sel_anchor_widget
        primary_item = (
            self.textblk_item_list[anchor.idx]
            if anchor is not None
            and self.textblk_item_list[anchor.idx] in selected
            else None
        )
        self.canvas.set_primary_selected_text_item(primary_item)
        # Refresh transform/format consumers without syncing back into the
        # pair list and discarding its Shift/drag anchor.
        self._update_selection_panels(selected, primary_item)

    def on_textedit_list_focusout(self):
        fw = self.app.focusWidget()
        focusing_edit = isinstance(fw, (SourceTextEdit, TransTextEdit))
        if fw == self.canvas.gv or focusing_edit:
            self.textEditList.clearDrag()
        if focusing_edit:
            self.textEditList.clearAllSelected()

    def on_rearrange_blks(self, mv_map: Tuple[np.ndarray]):
        self.canvas.push_undo_command(RearrangeBlksCommand(mv_map, self))

    def on_path_reorder_finished(self, touched_ids: Sequence[int]) -> None:
        source_ids, target_ids = build_path_reorder_map(
            touched_ids,
            len(self.textblk_item_list),
        )
        if source_ids:
            self.on_rearrange_blks((source_ids, target_ids))

    def updateTextBlkItemIdx(self, sel_ids: set = None):
        for ii, blk_item in enumerate(self.textblk_item_list):
            if sel_ids is not None and ii not in sel_ids:
                continue
            if blk_item.idx != ii:
                blk_item.idx = ii
                blk_item.refresh_order_badge()
                blk_item.update()
            self.pairwidget_list[ii].updateIndex(ii)
        cl = self.textEditList.checked_list
        if len(cl) != 0:
            cl.sort(key=lambda x: x.idx)

    def updateTextBlkList(self) -> None:
        self.canvas.text_move_session.cancel()
        cbl = self.imgtrans_proj.current_block_list()
        if cbl is None:
            return
        cbl.clear()
        for blk_item, trans_pair in zip(self.textblk_item_list, self.pairwidget_list):
            if not blk_item.document().isEmpty():
                blk_item.blk.rich_text = blk_item.toHtml()
                blk_item.blk.translation = blk_item.toPlainText()
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            blk_item.blk._bounding_rect = blk_item.absBoundingRect()
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def showTextblkItemRect(self, draw_rect: bool):
        self.canvas.textblock_mode = bool(draw_rect)
        for blk_item in self.textblk_item_list:
            blk_item.draw_rect = bool(draw_rect)
            blk_item.update()

    def set_blkitems_selection(self, selected: bool, blk_items: List[TextBlkItem] = None):
        self.canvas.block_selection_signal = True
        if blk_items is None:
            blk_items = self.textblk_item_list
        for blk_item in blk_items:
            blk_item.setSelected(selected)
        self.canvas.block_selection_signal = False
        self.on_incanvas_selection_changed()

    def on_ensure_textitem_svisible(self):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        self.changeHoveringWidget(edit)
        self.canvas.gv.ensureVisible(self.textblk_item_list[edit.idx])
        self.txtblkShapeControl.setBlkItem(self.textblk_item_list[edit.idx])

    def on_page_replace_one(self):
        self.canvas.push_undo_command(PageReplaceOneCommand(self.canvas.search_widget))

    def on_page_replace_all(self):
        self.canvas.push_undo_command(PageReplaceAllCommand(self.canvas.search_widget))

def get_text_size(fm: QFontMetricsF, text: str) -> Tuple[int, int]:
    brt = fm.tightBoundingRect(text)
    return int(np.ceil(fm.horizontalAdvance(text))), int(np.ceil(brt.height()))
    
def get_words_length_list(fm: QFontMetricsF, words: List[str]) -> List[int]:
    return [int(np.ceil(fm.horizontalAdvance(word))) for word in words]
